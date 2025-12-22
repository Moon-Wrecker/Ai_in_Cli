"""
Semantic Indexer - Embedding-based code indexing using OpenAI
Creates vector representations for semantic search
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from storage.chroma_store import ChromaVectorStore, get_vector_store
from utils.parsers import CodeChunker
from indexing.ast_indexer import ASTIndexer


@dataclass
class SemanticIndexStats:
    """Statistics from semantic indexing"""
    files_processed: int = 0
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    total_tokens_estimated: int = 0
    errors: List[str] = None
    duration_seconds: float = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "chunks_indexed": self.chunks_indexed,
            "chunks_skipped": self.chunks_skipped,
            "total_tokens_estimated": self.total_tokens_estimated,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class SemanticIndexer:
    """
    Creates semantic embeddings for code chunks using OpenAI.
    Stores vectors in ChromaDB for similarity search.
    """
    
    def __init__(
        self,
        sandbox_path: Optional[Path] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.vector_store = vector_store or get_vector_store()
        self.settings = get_settings()
        
        self.ast_indexer = ASTIndexer(sandbox_path=self.sandbox_path)
        self.chunker = CodeChunker(
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        
        # Track indexed files
        self.indexed_files: Set[str] = set()
    
    def index_workspace(self, force: bool = False) -> SemanticIndexStats:
        """
        Index all supported files in the sandbox with semantic embeddings.
        
        Args:
            force: If True, re-index all files
            
        Returns:
            SemanticIndexStats with results
        """
        start_time = time.time()
        stats = SemanticIndexStats()
        
        # Find all indexable files
        files = self._find_indexable_files()
        
        for file_path in files:
            try:
                # Index the file
                file_stats = self.index_file(file_path, force=force)
                stats.files_processed += 1
                stats.chunks_indexed += file_stats.get("chunks_added", 0)
                stats.chunks_skipped += file_stats.get("chunks_skipped", 0)
                stats.total_tokens_estimated += file_stats.get("tokens", 0)
                
                if file_stats.get("error"):
                    stats.errors.append(file_stats["error"])
                    
            except Exception as e:
                stats.errors.append(f"Error indexing {file_path}: {str(e)}")
        
        stats.duration_seconds = time.time() - start_time
        return stats
    
    def index_file(self, file_path: Path, force: bool = False) -> Dict[str, Any]:
        """
        Index a single file with semantic embeddings.
        
        Returns:
            Dictionary with indexing results
        """
        result = {
            "file": str(file_path),
            "chunks_added": 0,
            "chunks_skipped": 0,
            "tokens": 0,
            "error": None,
        }
        
        # Check if already indexed (unless force)
        if not force and str(file_path) in self.indexed_files:
            result["chunks_skipped"] = 1
            return result
        
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            result["error"] = f"Could not read file: {e}"
            return result
        
        # Delete existing chunks for this file if re-indexing
        if force:
            self.vector_store.delete_file_chunks(str(file_path))
        
        # Create chunks based on file type
        if file_path.suffix == ".py":
            chunks = self.ast_indexer.get_chunks(file_path)
        else:
            chunks = self._simple_chunk(content, str(file_path))
        
        if not chunks:
            return result
        
        # Enrich chunks with additional context
        enriched_chunks = self._enrich_chunks(chunks, file_path)
        
        # Estimate tokens
        for chunk in enriched_chunks:
            # Rough estimate: ~4 characters per token
            result["tokens"] += len(chunk.get("content", "")) // 4
        
        # Add to vector store
        added = self.vector_store.add_chunks(enriched_chunks)
        result["chunks_added"] = added
        
        self.indexed_files.add(str(file_path))
        
        return result
    
    def _find_indexable_files(self) -> List[Path]:
        """Find all files that can be indexed"""
        indexable = []
        allowed_extensions = self.settings.allowed_extensions
        ignore_patterns = self.settings.ignore_patterns
        
        for root, dirs, files in self.sandbox_path.rglob("*"):
            pass  # Using rglob instead
        
        for file_path in self.sandbox_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check extension
            if file_path.suffix.lower() not in allowed_extensions:
                continue
            
            # Check ignore patterns
            skip = False
            for pattern in ignore_patterns:
                if pattern in str(file_path):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Check file size
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self.settings.max_file_size_mb:
                    continue
            except:
                continue
            
            indexable.append(file_path)
        
        return indexable
    
    def _simple_chunk(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Create simple line-based chunks for non-Python files"""
        chunks = []
        lines = content.split("\n")
        
        chunk_lines = self.settings.chunk_size // 50  # ~50 chars per line
        overlap_lines = self.settings.chunk_overlap // 50
        
        start = 0
        while start < len(lines):
            end = min(start + chunk_lines, len(lines))
            chunk_content = "\n".join(lines[start:end])
            
            if chunk_content.strip():  # Skip empty chunks
                chunks.append({
                    "content": chunk_content,
                    "file": file_path,
                    "start_line": start + 1,
                    "end_line": end,
                    "type": "text",
                    "context": None,
                })
            
            start = end - overlap_lines
            if start >= len(lines) - overlap_lines:
                break
        
        return chunks
    
    def _enrich_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_path: Path,
    ) -> List[Dict[str, Any]]:
        """
        Enrich chunks with additional context for better retrieval.
        Adds file context, parent info, and semantic hints.
        """
        enriched = []
        
        # Get file-level context
        file_context = self._get_file_context(file_path)
        
        for chunk in chunks:
            enriched_content = self._build_enriched_content(chunk, file_context)
            
            enriched_chunk = {
                **chunk,
                "content": enriched_content,
                "original_content": chunk["content"],
            }
            
            enriched.append(enriched_chunk)
        
        return enriched
    
    def _get_file_context(self, file_path: Path) -> Dict[str, Any]:
        """Get context information about a file"""
        context = {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "relative_path": str(file_path.relative_to(self.sandbox_path)),
        }
        
        # Add AST info for Python files
        if file_path.suffix == ".py":
            try:
                structure = self.ast_indexer.get_file_structure(file_path)
                context["classes"] = [c["name"] for c in structure.get("classes", [])]
                context["functions"] = [f["name"] for f in structure.get("functions", [])]
                context["docstring"] = structure.get("module_docstring", "")
            except:
                pass
        
        return context
    
    def _build_enriched_content(
        self,
        chunk: Dict[str, Any],
        file_context: Dict[str, Any],
    ) -> str:
        """Build enriched content string for embedding"""
        parts = []
        
        # Add file context
        parts.append(f"File: {file_context['relative_path']}")
        
        # Add chunk location
        parts.append(f"Lines {chunk['start_line']}-{chunk['end_line']}")
        
        # Add type info
        chunk_type = chunk.get("type", "text")
        if chunk_type != "text":
            parts.append(f"Type: {chunk_type}")
        
        # Add symbol context if available
        context = chunk.get("context")
        if context:
            if context.get("name"):
                parts.append(f"Symbol: {context['name']}")
            if context.get("signature"):
                parts.append(f"Signature: {context['signature']}")
            if context.get("docstring"):
                # Truncate long docstrings
                doc = context["docstring"]
                if len(doc) > 200:
                    doc = doc[:200] + "..."
                parts.append(f"Doc: {doc}")
        
        # Add the actual content
        parts.append("")  # Empty line separator
        parts.append(chunk["content"])
        
        return "\n".join(parts)
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        file_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar code chunks.
        
        Args:
            query: Search query
            n_results: Maximum results
            file_filter: Filter by file path
            type_filter: Filter by chunk type
            
        Returns:
            List of search results
        """
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_file=file_filter,
            filter_type=type_filter,
        )
        
        return [r.to_dict() for r in results]
    
    def get_file_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all indexed chunks for a file"""
        results = self.vector_store.search_by_file(file_path)
        return [r.to_dict() for r in results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        store_stats = self.vector_store.get_stats()
        
        return {
            "indexed_files": len(self.indexed_files),
            "vector_store": store_stats,
        }
    
    def clear_index(self):
        """Clear all indexed data"""
        self.vector_store.clear()
        self.indexed_files.clear()


# Convenience function
def index_semantic(force: bool = False) -> SemanticIndexStats:
    """Index the sandbox with semantic embeddings"""
    indexer = SemanticIndexer()
    return indexer.index_workspace(force=force)



