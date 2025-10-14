"""
RAG (Retrieval-Augmented Generation) System for Workspace Awareness
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import time

# Suppress ChromaDB warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Chroma.*')

try:
    from chromadb import Client
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    Client = None
    Settings = None
    embedding_functions = None

@dataclass
class DocumentChunk:
    """Represents a chunk of code or text"""
    id: str
    content: str
    file_path: str
    chunk_type: str  # 'function', 'class', 'text', 'config'
    language: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any]

class WorkspaceRAG:
    """RAG system for workspace awareness and code understanding"""
    
    def __init__(self, workspace_dir: str, persist_dir: str = None):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.persist_dir = Path(persist_dir) if persist_dir else (self.workspace_dir / ".rag_cache")
        self.persist_dir.mkdir(exist_ok=True)
        
        self.index_file = self.persist_dir / "file_index.json"
        self.chunks_file = self.persist_dir / "chunks.json"
        
        # File extensions to process
        self.text_extensions = {
            '.py', '.js', '.ts', '.html', '.css', '.md', '.txt', 
            '.json', '.yml', '.yaml', '.xml', '.sql', '.sh', '.bat',
            '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go'
        }
        
        # Initialize ChromaDB if available
        self.chroma_client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            try:
                # Suppress stdout warnings from ChromaDB
                import sys
                from io import StringIO
                old_stderr = sys.stderr
                sys.stderr = StringIO()
                
                self._init_chromadb()
                
                sys.stderr = old_stderr
            except Exception as e:
                sys.stderr = old_stderr
                # Silently fail and use fallback
                pass
        
        # File index for tracking changes
        self.file_index = self._load_file_index()
        self.chunks = self._load_chunks()
    
    def _init_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        if not CHROMADB_AVAILABLE or Client is None:
            return
            
        chroma_persist_dir = str(self.persist_dir / "chroma_db")
        self.chroma_client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_persist_dir
        ))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("workspace_code")
        except:
            # Create collection with default embedding function
            if embedding_functions:
                embedding_function = embedding_functions.DefaultEmbeddingFunction()
                self.collection = self.chroma_client.create_collection(
                    name="workspace_code",
                    embedding_function=embedding_function
                )
    
    def _load_file_index(self) -> Dict[str, Dict[str, Any]]:
        """Load file index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_file_index(self):
        """Save file index to disk"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_index, f, indent=2)
    
    def _load_chunks(self) -> Dict[str, DocumentChunk]:
        """Load chunks from disk"""
        if self.chunks_file.exists():
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                chunks = {}
                for chunk_id, data in chunk_data.items():
                    chunks[chunk_id] = DocumentChunk(**data)
                return chunks
        return {}
    
    def _save_chunks(self):
        """Save chunks to disk"""
        chunk_data = {}
        for chunk_id, chunk in self.chunks.items():
            chunk_data[chunk_id] = asdict(chunk)
        
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        if not file_path.is_file():
            return False
        
        # Check extension
        if file_path.suffix.lower() not in self.text_extensions:
            return False
        
        # Skip hidden files and cache directories
        if any(part.startswith('.') for part in file_path.parts):
            if '.rag_cache' not in str(file_path):  # Allow our own cache
                return False
        
        # Skip large files (>1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except:
            return False
        
        return True
    
    def _chunk_python_file(self, file_path: Path, content: str) -> List[DocumentChunk]:
        """Chunk Python file into logical units"""
        chunks = []
        lines = content.splitlines()
        
        # Track current function/class
        current_function = None
        current_class = None
        chunk_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Class definition
            if stripped.startswith('class '):
                if current_function or current_class:
                    # Save previous chunk
                    chunk_content = '\n'.join(lines[chunk_start:i])
                    if chunk_content.strip():
                        chunks.append(self._create_chunk(
                            file_path, chunk_content, chunk_start + 1, i,
                            'class' if current_class else 'function',
                            {'name': current_class or current_function}
                        ))
                
                class_match = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                current_class = class_match
                current_function = None
                chunk_start = i
            
            # Function definition
            elif stripped.startswith('def '):
                if current_function and not current_class:
                    # Save previous function (not inside a class)
                    chunk_content = '\n'.join(lines[chunk_start:i])
                    if chunk_content.strip():
                        chunks.append(self._create_chunk(
                            file_path, chunk_content, chunk_start + 1, i,
                            'function', {'name': current_function}
                        ))
                    chunk_start = i
                
                func_match = line.split('def ')[1].split('(')[0].strip()
                current_function = func_match
        
        # Save final chunk
        if chunk_start < len(lines):
            chunk_content = '\n'.join(lines[chunk_start:])
            if chunk_content.strip():
                chunk_type = 'class' if current_class else 'function' if current_function else 'text'
                name = current_class or current_function or 'end_of_file'
                chunks.append(self._create_chunk(
                    file_path, chunk_content, chunk_start + 1, len(lines),
                    chunk_type, {'name': name}
                ))
        
        return chunks
    
    def _chunk_text_file(self, file_path: Path, content: str) -> List[DocumentChunk]:
        """Chunk non-code files"""
        chunks = []
        lines = content.splitlines()
        
        # For text files, create chunks of ~50 lines
        chunk_size = 50
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            if chunk_content.strip():
                chunks.append(self._create_chunk(
                    file_path, chunk_content, i + 1, min(i + chunk_size, len(lines)),
                    'text', {'chunk_number': i // chunk_size + 1}
                ))
        
        return chunks
    
    def _create_chunk(self, file_path: Path, content: str, start_line: int, end_line: int, 
                     chunk_type: str, metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a document chunk"""
        rel_path = file_path.relative_to(self.workspace_dir)
        chunk_id = f"{rel_path}:{start_line}-{end_line}:{chunk_type}"
        
        language = self._detect_language(file_path)
        
        metadata.update({
            'file_size': file_path.stat().st_size,
            'modified_time': file_path.stat().st_mtime,
            'lines_count': end_line - start_line + 1
        })
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            file_path=str(rel_path),
            chunk_type=chunk_type,
            language=language,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata
        )
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.xml': 'xml',
            '.sql': 'sql',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go'
        }
        return ext_map.get(file_path.suffix.lower(), 'text')
    
    def index_workspace(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Index all files in workspace"""
        stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'chunks_created': 0,
            'errors': []
        }
        
        for file_path in self.workspace_dir.rglob('*'):
            if not self._should_process_file(file_path):
                continue
            
            try:
                rel_path = str(file_path.relative_to(self.workspace_dir))
                file_hash = self._get_file_hash(file_path)
                
                # Check if file changed
                if not force_reindex and rel_path in self.file_index:
                    if self.file_index[rel_path].get('hash') == file_hash:
                        stats['files_skipped'] += 1
                        continue
                
                # Process file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Create chunks
                if file_path.suffix.lower() == '.py':
                    chunks = self._chunk_python_file(file_path, content)
                else:
                    chunks = self._chunk_text_file(file_path, content)
                
                # Remove old chunks for this file
                old_chunk_ids = [cid for cid, chunk in self.chunks.items() 
                               if chunk.file_path == rel_path]
                for cid in old_chunk_ids:
                    del self.chunks[cid]
                    if self.collection:
                        try:
                            self.collection.delete(ids=[cid])
                        except:
                            pass
                
                # Add new chunks
                for chunk in chunks:
                    self.chunks[chunk.id] = chunk
                    stats['chunks_created'] += 1
                    
                    # Add to vector database
                    if self.collection:
                        try:
                            self.collection.add(
                                ids=[chunk.id],
                                documents=[chunk.content],
                                metadatas=[{
                                    'file_path': chunk.file_path,
                                    'chunk_type': chunk.chunk_type,
                                    'language': chunk.language,
                                    'start_line': chunk.start_line,
                                    'end_line': chunk.end_line,
                                    **chunk.metadata
                                }]
                            )
                        except Exception as e:
                            stats['errors'].append(f"Vector DB error for {chunk.id}: {e}")
                
                # Update file index
                self.file_index[rel_path] = {
                    'hash': file_hash,
                    'processed_time': time.time(),
                    'chunk_count': len(chunks)
                }
                
                stats['files_processed'] += 1
                
            except Exception as e:
                stats['errors'].append(f"Error processing {file_path}: {e}")
        
        # Persist data
        self._save_file_index()
        self._save_chunks()
        if self.chroma_client:
            self.chroma_client.persist()
        
        return stats
    
    def search_code(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for code using semantic similarity"""
        results = []
        
        if self.collection:
            try:
                # Vector search
                vector_results = self.collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                for i, doc_id in enumerate(vector_results['ids'][0]):
                    if doc_id in self.chunks:
                        chunk = self.chunks[doc_id]
                        results.append({
                            'chunk_id': doc_id,
                            'file_path': chunk.file_path,
                            'content': chunk.content,
                            'chunk_type': chunk.chunk_type,
                            'language': chunk.language,
                            'start_line': chunk.start_line,
                            'end_line': chunk.end_line,
                            'score': vector_results['distances'][0][i],
                            'metadata': chunk.metadata
                        })
            except Exception as e:
                print(f"Vector search failed: {e}")
        
        # Fallback to text search if vector search not available
        if not results:
            query_lower = query.lower()
            for chunk in self.chunks.values():
                if query_lower in chunk.content.lower():
                    results.append({
                        'chunk_id': chunk.id,
                        'file_path': chunk.file_path,
                        'content': chunk.content,
                        'chunk_type': chunk.chunk_type,
                        'language': chunk.language,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'score': 0.0,
                        'metadata': chunk.metadata
                    })
                    if len(results) >= limit:
                        break
        
        return results
    
    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive context for a specific file"""
        file_chunks = [chunk for chunk in self.chunks.values() 
                      if chunk.file_path == file_path]
        
        if not file_chunks:
            return {'error': f'No information found for {file_path}'}
        
        context = {
            'file_path': file_path,
            'language': file_chunks[0].language,
            'total_chunks': len(file_chunks),
            'functions': [],
            'classes': [],
            'structure': []
        }
        
        for chunk in sorted(file_chunks, key=lambda x: x.start_line):
            if chunk.chunk_type == 'function':
                context['functions'].append({
                    'name': chunk.metadata.get('name', 'unknown'),
                    'start_line': chunk.start_line,
                    'content': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                })
            elif chunk.chunk_type == 'class':
                context['classes'].append({
                    'name': chunk.metadata.get('name', 'unknown'),
                    'start_line': chunk.start_line,
                    'content': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                })
            
            context['structure'].append({
                'type': chunk.chunk_type,
                'name': chunk.metadata.get('name', 'text'),
                'lines': f"{chunk.start_line}-{chunk.end_line}"
            })
        
        return context
    
    def get_workspace_overview(self) -> Dict[str, Any]:
        """Get overview of the entire workspace"""
        overview = {
            'total_files': len(self.file_index),
            'total_chunks': len(self.chunks),
            'languages': {},
            'file_types': {},
            'recent_files': [],
            'largest_files': []
        }
        
        # Analyze chunks
        for chunk in self.chunks.values():
            # Language distribution
            lang = chunk.language
            overview['languages'][lang] = overview['languages'].get(lang, 0) + 1
            
            # File type distribution
            ftype = chunk.chunk_type
            overview['file_types'][ftype] = overview['file_types'].get(ftype, 0) + 1
        
        # Recent files (by modification time)
        file_times = []
        for file_path, info in self.file_index.items():
            file_times.append((file_path, info.get('processed_time', 0)))
        
        file_times.sort(key=lambda x: x[1], reverse=True)
        overview['recent_files'] = [f[0] for f in file_times[:10]]
        
        # Largest files (by chunk count)
        file_chunks = {}
        for chunk in self.chunks.values():
            fp = chunk.file_path
            file_chunks[fp] = file_chunks.get(fp, 0) + 1
        
        largest = sorted(file_chunks.items(), key=lambda x: x[1], reverse=True)
        overview['largest_files'] = largest[:10]
        
        return overview
    
    def get_related_code(self, file_path: str, context_window: int = 3) -> List[Dict[str, Any]]:
        """Get code related to a specific file"""
        # Get functions/classes from the target file
        target_chunks = [chunk for chunk in self.chunks.values() 
                        if chunk.file_path == file_path]
        
        related = []
        for target_chunk in target_chunks:
            if target_chunk.chunk_type in ['function', 'class']:
                name = target_chunk.metadata.get('name', '')
                if name:
                    # Search for references to this name in other files
                    results = self.search_code(name, limit=5)
                    for result in results:
                        if result['file_path'] != file_path:
                            related.append(result)
        
        return related[:10]  # Limit results
