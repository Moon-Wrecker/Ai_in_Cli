"""
Index Manager - Smart incremental indexing system
Tracks file changes and maintains index freshness automatically
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path, get_cache_path


@dataclass
class FileState:
    """Tracks the state of an indexed file"""
    path: str
    mtime: float  # Last modification time
    size: int
    indexed_at: float  # When we last indexed it
    checksum: Optional[str] = None  # Optional content hash
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        return cls(**data)


@dataclass  
class IndexState:
    """Persistent state of the entire index"""
    files: Dict[str, FileState] = field(default_factory=dict)
    last_full_index: float = 0
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "last_full_index": self.last_full_index,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexState":
        files = {
            k: FileState.from_dict(v) 
            for k, v in data.get("files", {}).items()
        }
        return cls(
            files=files,
            last_full_index=data.get("last_full_index", 0),
            version=data.get("version", "1.0"),
        )


class IndexManager:
    """
    Manages smart incremental indexing.
    
    Features:
    - Tracks file modification times
    - Detects new, modified, and deleted files
    - Provides hooks for auto-indexing after file operations
    - Supports incremental startup indexing
    """
    
    _instance: Optional["IndexManager"] = None
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.settings = get_settings()
        self.cache_path = get_cache_path()
        self.state_file = self.cache_path / "index_state.json"
        
        # Load persistent state
        self.state = self._load_state()
        
        # Lazy-loaded indexers
        self._ast_indexer = None
        self._semantic_indexer = None
        self._graph_builder = None
        
        # Directories to skip
        self.skip_dirs = {
            '.venv', 'venv', '__pycache__', '.git', 'node_modules',
            '.cache', '.chroma', 'chroma_data', '.mypy_cache', 
            '.pytest_cache', 'site-packages', '.chroma_db'
        }
    
    @classmethod
    def get_instance(cls, sandbox_path: Optional[Path] = None) -> "IndexManager":
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls(sandbox_path)
        return cls._instance
    
    def _load_state(self) -> IndexState:
        """Load index state from disk"""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return IndexState.from_dict(data)
            except Exception:
                pass
        return IndexState()
    
    def _save_state(self):
        """Save index state to disk"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as e:
            print(f"Warning: Could not save index state: {e}")
    
    @property
    def ast_indexer(self):
        """Lazy load AST indexer"""
        if self._ast_indexer is None:
            from indexing.ast_indexer import ASTIndexer
            self._ast_indexer = ASTIndexer(self.sandbox_path)
        return self._ast_indexer
    
    @property
    def semantic_indexer(self):
        """Lazy load semantic indexer"""
        if self._semantic_indexer is None:
            from indexing.semantic_indexer import SemanticIndexer
            self._semantic_indexer = SemanticIndexer(self.sandbox_path)
        return self._semantic_indexer
    
    @property
    def graph_builder(self):
        """Lazy load graph builder"""
        if self._graph_builder is None:
            from indexing.dependency_graph import DependencyGraphBuilder
            self._graph_builder = DependencyGraphBuilder(self.sandbox_path)
        return self._graph_builder
    
    def get_file_state(self, file_path: Path) -> Optional[FileState]:
        """Get current state of a file on disk"""
        try:
            if not file_path.exists():
                return None
            stat = file_path.stat()
            return FileState(
                path=str(file_path),
                mtime=stat.st_mtime,
                size=stat.st_size,
                indexed_at=0,
            )
        except Exception:
            return None
    
    def is_file_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last index"""
        path_str = str(file_path.resolve())
        
        # Not indexed before
        if path_str not in self.state.files:
            return True
        
        current = self.get_file_state(file_path)
        if current is None:
            return True  # File deleted
        
        stored = self.state.files[path_str]
        
        # Check modification time and size
        return (
            current.mtime > stored.indexed_at or
            current.size != stored.size
        )
    
    def get_changed_files(self) -> Dict[str, List[Path]]:
        """
        Scan workspace and categorize files by change status.
        
        Returns:
            Dict with keys: 'new', 'modified', 'deleted'
        """
        result = {
            "new": [],
            "modified": [],
            "deleted": [],
        }
        
        # Get all current Python files
        current_files: Set[str] = set()
        
        for file_path in self.sandbox_path.rglob("*.py"):
            # Skip ignored directories
            if any(skip in file_path.parts for skip in self.skip_dirs):
                continue
            
            path_str = str(file_path.resolve())
            current_files.add(path_str)
            
            if path_str not in self.state.files:
                result["new"].append(file_path)
            elif self.is_file_changed(file_path):
                result["modified"].append(file_path)
        
        # Find deleted files
        for path_str in self.state.files:
            if path_str not in current_files:
                result["deleted"].append(Path(path_str))
        
        return result
    
    def index_file(self, file_path: Path, reason: str = "manual") -> Dict[str, Any]:
        """
        Index a single file and update state.
        
        Args:
            file_path: Path to file
            reason: Why indexing (for logging)
            
        Returns:
            Dict with indexing results
        """
        result = {
            "file": str(file_path),
            "reason": reason,
            "success": False,
            "ast_symbols": 0,
            "semantic_chunks": 0,
            "errors": [],
        }
        
        resolved = file_path.resolve()
        if not resolved.exists():
            result["errors"].append("File does not exist")
            return result
        
        # Only index Python files for now
        if resolved.suffix != ".py":
            result["errors"].append("Not a Python file (skipped)")
            result["success"] = True  # Not an error, just skipped
            return result
        
        try:
            # AST indexing
            ast_result = self.ast_indexer.index_file(resolved)
            result["ast_symbols"] = ast_result.get("symbols", 0)
            
            # Semantic indexing
            sem_result = self.semantic_indexer.index_file(resolved)
            result["semantic_chunks"] = sem_result.get("chunks_indexed", 0)
            
            # Update state
            current = self.get_file_state(resolved)
            if current:
                current.indexed_at = time.time()
                self.state.files[str(resolved)] = current
                self._save_state()
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def remove_file_from_index(self, file_path: Path) -> Dict[str, Any]:
        """
        Remove a file from the index.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict with removal results
        """
        result = {
            "file": str(file_path),
            "removed": False,
        }
        
        path_str = str(file_path.resolve())
        
        # Remove from state
        if path_str in self.state.files:
            del self.state.files[path_str]
            self._save_state()
            result["removed"] = True
        
        # Remove from graph
        try:
            from storage.graph_store import get_dependency_graph
            graph = get_dependency_graph()
            graph.remove_file(path_str)
        except Exception:
            pass
        
        # Note: ChromaDB doesn't easily support deletion by metadata
        # The stale chunks will be overwritten on next full index
        
        return result
    
    def incremental_index(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform incremental indexing - only index changed files.
        
        Returns:
            Dict with indexing statistics
        """
        start_time = time.time()
        
        result = {
            "mode": "incremental",
            "new_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "total_symbols": 0,
            "total_chunks": 0,
            "errors": [],
            "duration_seconds": 0,
        }
        
        changes = self.get_changed_files()
        
        # Index new files
        for file_path in changes["new"]:
            if verbose:
                print(f"  + Indexing new: {file_path.name}")
            idx_result = self.index_file(file_path, reason="new")
            if idx_result["success"]:
                result["new_files"] += 1
                result["total_symbols"] += idx_result["ast_symbols"]
                result["total_chunks"] += idx_result["semantic_chunks"]
            else:
                result["errors"].extend(idx_result["errors"])
        
        # Index modified files
        for file_path in changes["modified"]:
            if verbose:
                print(f"  ~ Indexing modified: {file_path.name}")
            idx_result = self.index_file(file_path, reason="modified")
            if idx_result["success"]:
                result["modified_files"] += 1
                result["total_symbols"] += idx_result["ast_symbols"]
                result["total_chunks"] += idx_result["semantic_chunks"]
            else:
                result["errors"].extend(idx_result["errors"])
        
        # Handle deleted files
        for file_path in changes["deleted"]:
            if verbose:
                print(f"  - Removing deleted: {file_path.name}")
            self.remove_file_from_index(file_path)
            result["deleted_files"] += 1
        
        result["duration_seconds"] = round(time.time() - start_time, 2)
        
        return result
    
    def full_index(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform full workspace indexing.
        
        Returns:
            Dict with indexing statistics
        """
        start_time = time.time()
        
        result = {
            "mode": "full",
            "files_indexed": 0,
            "total_symbols": 0,
            "total_chunks": 0,
            "errors": [],
            "duration_seconds": 0,
        }
        
        # Clear existing state
        self.state = IndexState()
        
        # Find all Python files
        for file_path in self.sandbox_path.rglob("*.py"):
            if any(skip in file_path.parts for skip in self.skip_dirs):
                continue
            
            if verbose:
                print(f"  Indexing: {file_path.name}")
            
            idx_result = self.index_file(file_path, reason="full")
            if idx_result["success"]:
                result["files_indexed"] += 1
                result["total_symbols"] += idx_result["ast_symbols"]
                result["total_chunks"] += idx_result["semantic_chunks"]
            else:
                result["errors"].extend(idx_result["errors"])
        
        self.state.last_full_index = time.time()
        self._save_state()
        
        result["duration_seconds"] = round(time.time() - start_time, 2)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        return {
            "indexed_files": len(self.state.files),
            "last_full_index": datetime.fromtimestamp(
                self.state.last_full_index
            ).isoformat() if self.state.last_full_index else None,
            "state_file": str(self.state_file),
        }
    
    def needs_indexing(self) -> bool:
        """Check if workspace needs indexing"""
        changes = self.get_changed_files()
        return bool(changes["new"] or changes["modified"] or changes["deleted"])


# Singleton accessor
def get_index_manager(sandbox_path: Optional[Path] = None) -> IndexManager:
    """Get the global IndexManager instance"""
    return IndexManager.get_instance(sandbox_path)


# Convenience function for file operation hooks
def on_file_created(file_path: Path) -> Dict[str, Any]:
    """Hook called when a file is created"""
    manager = get_index_manager()
    return manager.index_file(file_path, reason="created")


def on_file_modified(file_path: Path) -> Dict[str, Any]:
    """Hook called when a file is modified"""
    manager = get_index_manager()
    return manager.index_file(file_path, reason="modified")


def on_file_deleted(file_path: Path) -> Dict[str, Any]:
    """Hook called when a file is deleted"""
    manager = get_index_manager()
    return manager.remove_file_from_index(file_path)

