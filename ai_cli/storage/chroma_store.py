"""
ChromaDB Vector Store for semantic search
Provides in-memory and persistent vector storage with OpenAI embeddings
"""

import os
import hashlib
import time
import warnings
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Suppress ChromaDB telemetry errors BEFORE any chromadb imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"
warnings.filterwarnings("ignore", message=".*telemetry.*")

# Create a mock posthog module to prevent telemetry completely
class MockPosthog:
    def __init__(self, *args, **kwargs): pass
    def capture(self, *args, **kwargs): pass
    def identify(self, *args, **kwargs): pass
    def alias(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass
    def group(self, *args, **kwargs): pass
    def feature_enabled(self, *args, **kwargs): return False
    def shutdown(self, *args, **kwargs): pass
    api_key = None
    disabled = True

# Patch posthog before chromadb imports it
sys.modules['posthog'] = MockPosthog()

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings, get_chroma_path


@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    chunk_type: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "file": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": self.score,
            "type": self.chunk_type,
            "metadata": self.metadata,
        }


class OpenAIEmbeddings:
    """
    OpenAI embeddings wrapper for ChromaDB.
    Uses text-embedding-3-small by default for efficiency.
    """
    
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.openai_embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        self._cache: Dict[str, List[float]] = {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._embed_batch([text])[0]
        self._cache[cache_key] = result
        return result
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts"""
        # Clean texts (remove empty strings, limit length)
        cleaned = []
        for text in texts:
            text = text.strip()
            if not text:
                text = " "  # ChromaDB requires non-empty
            # Truncate to ~8000 tokens (rough estimate: 4 chars per token)
            if len(text) > 32000:
                text = text[:32000]
            cleaned.append(text)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Embedding error: {e}")
            # Return zero vectors on error
            return [[0.0] * 1536 for _ in cleaned]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()


class ChromaVectorStore:
    """
    ChromaDB-based vector store for code chunks.
    Supports both in-memory and persistent storage.
    """
    
    def __init__(
        self,
        collection_name: str = "code_chunks",
        persist: bool = True,
        persist_path: Optional[Path] = None,
    ):
        self.collection_name = collection_name
        self.persist = persist
        self.persist_path = persist_path or get_chroma_path()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB
        self._init_client()
        self._init_collection()
    
    def _init_client(self):
        """Initialize ChromaDB client"""
        if self.persist:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        else:
            self.client = chromadb.Client(
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
    
    def _init_collection(self):
        """Initialize or get collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Try to reset and recreate
            self.client.reset()
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add code chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content', 'file', etc.
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Prepare data
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = self._generate_chunk_id(chunk)
            
            # Skip if already exists
            if self._chunk_exists(chunk_id):
                continue
            
            ids.append(chunk_id)
            documents.append(chunk["content"])
            
            # Prepare metadata (ChromaDB has type restrictions)
            metadata = {
                "file": chunk.get("file", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "type": chunk.get("type", "text"),
                "indexed_at": time.time(),
            }
            
            # Add context if present (serialize as JSON string)
            if chunk.get("context"):
                metadata["context"] = json.dumps(chunk["context"])
            
            metadatas.append(metadata)
        
        if not ids:
            return 0
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(documents)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as e:
            print(f"Error adding chunks: {e}")
            return 0
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_file: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Semantic search for relevant code chunks.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            filter_file: Optional file path filter
            filter_type: Optional chunk type filter
            
        Returns:
            List of SearchResult objects
        """
        # Build filter
        where_filter = None
        if filter_file or filter_type:
            conditions = []
            if filter_file:
                conditions.append({"file": {"$eq": filter_file}})
            if filter_type:
                conditions.append({"type": {"$eq": filter_type}})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        # Parse results
        search_results = []
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        for i, chunk_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity score (cosine distance to similarity)
            score = 1 - distance
            
            # Parse context if present
            context = {}
            if "context" in metadata:
                try:
                    context = json.loads(metadata["context"])
                except:
                    pass
            
            search_results.append(SearchResult(
                content=doc,
                file_path=metadata.get("file", ""),
                start_line=metadata.get("start_line", 0),
                end_line=metadata.get("end_line", 0),
                score=score,
                chunk_type=metadata.get("type", "text"),
                metadata=context,
            ))
        
        return search_results
    
    def search_by_file(self, file_path: str) -> List[SearchResult]:
        """Get all chunks for a specific file"""
        try:
            results = self.collection.get(
                where={"file": {"$eq": file_path}},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            print(f"Error getting file chunks: {e}")
            return []
        
        search_results = []
        for i, chunk_id in enumerate(results["ids"]):
            doc = results["documents"][i]
            metadata = results["metadatas"][i]
            
            context = {}
            if "context" in metadata:
                try:
                    context = json.loads(metadata["context"])
                except:
                    pass
            
            search_results.append(SearchResult(
                content=doc,
                file_path=metadata.get("file", ""),
                start_line=metadata.get("start_line", 0),
                end_line=metadata.get("end_line", 0),
                score=1.0,  # Exact match
                chunk_type=metadata.get("type", "text"),
                metadata=context,
            ))
        
        # Sort by line number
        search_results.sort(key=lambda x: x.start_line)
        return search_results
    
    def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks for a file (for re-indexing)"""
        try:
            # Get IDs of chunks for this file
            results = self.collection.get(
                where={"file": {"$eq": file_path}},
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception as e:
            print(f"Error deleting file chunks: {e}")
        
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            # Get file distribution
            results = self.collection.get(include=["metadatas"])
            files = set()
            types = {}
            
            for metadata in results["metadatas"]:
                files.add(metadata.get("file", ""))
                chunk_type = metadata.get("type", "unknown")
                types[chunk_type] = types.get(chunk_type, 0) + 1
            
            return {
                "total_chunks": count,
                "total_files": len(files),
                "files": list(files),
                "type_distribution": types,
                "persist_path": str(self.persist_path) if self.persist else None,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def _generate_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """Generate unique ID for a chunk"""
        content_hash = hashlib.md5(chunk["content"].encode()).hexdigest()[:8]
        file_hash = hashlib.md5(chunk.get("file", "").encode()).hexdigest()[:8]
        return f"{file_hash}_{chunk.get('start_line', 0)}_{content_hash}"
    
    def _chunk_exists(self, chunk_id: str) -> bool:
        """Check if chunk already exists"""
        try:
            result = self.collection.get(ids=[chunk_id])
            return len(result["ids"]) > 0
        except:
            return False


# Singleton instance
_vector_store: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store



