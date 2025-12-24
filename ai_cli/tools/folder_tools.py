"""
Folder Tools - Directory operations
All operations are sandboxed to the sandbox directory
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from utils.security import PathValidator, SecurityError


class FolderTools:
    """
    Provides directory operations with security sandboxing.
    All operations are restricted to the sandbox directory.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.validator = PathValidator(self.sandbox_path)
        self.settings = get_settings()
    
    def list_directory(
        self,
        path: str = ".",
        recursive: bool = False,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path (relative to sandbox)
            recursive: If True, list recursively
            include_hidden: If True, include hidden files/folders
            
        Returns:
            Dict with directory listing
        """
        try:
            resolved = self.validator.resolve_path(path)
            
            if not resolved.exists():
                return {"error": f"Directory not found: {path}"}
            
            if not resolved.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            if recursive:
                return self._list_recursive(resolved, include_hidden)
            else:
                return self._list_flat(resolved, include_hidden)
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to list directory: {str(e)}"}
    
    def _list_flat(self, path: Path, include_hidden: bool) -> Dict[str, Any]:
        """List directory contents (non-recursive)"""
        files = []
        folders = []
        
        for item in sorted(path.iterdir()):
            # Skip hidden unless requested
            if not include_hidden and item.name.startswith("."):
                continue
            
            # Skip ignored patterns
            if any(p in item.name for p in self.settings.ignore_patterns):
                continue
            
            info = {
                "name": item.name,
                "path": str(item),
                "relative_path": self.validator.get_relative_path(item),
            }
            
            if item.is_file():
                stat = item.stat()
                info["size_bytes"] = stat.st_size
                info["extension"] = item.suffix
                files.append(info)
            else:
                folders.append(info)
        
        return {
            "path": str(path),
            "relative_path": self.validator.get_relative_path(path),
            "folders": folders,
            "files": files,
            "folder_count": len(folders),
            "file_count": len(files),
        }
    
    def _list_recursive(self, path: Path, include_hidden: bool) -> Dict[str, Any]:
        """List directory contents recursively as a tree"""
        
        def build_tree(current_path: Path, depth: int = 0) -> Dict[str, Any]:
            if depth > 10:  # Prevent too deep recursion
                return {"name": "...", "truncated": True}
            
            tree = {
                "name": current_path.name,
                "path": self.validator.get_relative_path(current_path),
                "type": "directory",
                "children": [],
            }
            
            try:
                items = sorted(current_path.iterdir())
            except PermissionError:
                tree["error"] = "Permission denied"
                return tree
            
            for item in items:
                if not include_hidden and item.name.startswith("."):
                    continue
                
                if any(p in item.name for p in self.settings.ignore_patterns):
                    continue
                
                if item.is_dir():
                    tree["children"].append(build_tree(item, depth + 1))
                else:
                    tree["children"].append({
                        "name": item.name,
                        "path": self.validator.get_relative_path(item),
                        "type": "file",
                        "size_bytes": item.stat().st_size,
                        "extension": item.suffix,
                    })
            
            return tree
        
        tree = build_tree(path)
        
        # Also create a flat text representation
        lines = []
        
        def render_tree(node: Dict, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            icon = "📁" if node.get("type") == "directory" else "📄"
            lines.append(f"{prefix}{connector}{icon} {node['name']}")
            
            children = node.get("children", [])
            for i, child in enumerate(children):
                extension = "    " if is_last else "│   "
                render_tree(child, prefix + extension, i == len(children) - 1)
        
        render_tree(tree, "", True)
        
        return {
            "path": str(path),
            "relative_path": self.validator.get_relative_path(path),
            "tree": tree,
            "tree_text": "\n".join(lines),
        }
    
    def create_directory(self, path: str) -> Dict[str, Any]:
        """
        Create a directory (and parent directories if needed).
        
        Args:
            path: Directory path to create
            
        Returns:
            Dict with creation result
        """
        try:
            resolved = self.validator.resolve_path(path)
            
            if resolved.exists():
                if resolved.is_dir():
                    return {
                        "success": True,
                        "path": str(resolved),
                        "relative_path": self.validator.get_relative_path(resolved),
                        "already_exists": True,
                    }
                else:
                    return {"error": f"A file exists at this path: {path}"}
            
            resolved.mkdir(parents=True, exist_ok=True)
            
            return {
                "success": True,
                "path": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "created": True,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to create directory: {str(e)}"}
    
    def delete_directory(
        self,
        path: str,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a directory.
        
        Args:
            path: Directory path to delete
            recursive: If True, delete contents recursively
            
        Returns:
            Dict with deletion result
        """
        try:
            resolved = self.validator.resolve_path(path)
            
            if not resolved.exists():
                return {"error": f"Directory not found: {path}"}
            
            if not resolved.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            # Check if directory is empty
            contents = list(resolved.iterdir())
            
            if contents and not recursive:
                return {
                    "error": f"Directory not empty. Use recursive=True to delete all contents.",
                    "item_count": len(contents),
                }
            
            if recursive:
                shutil.rmtree(resolved)
            else:
                resolved.rmdir()
            
            return {
                "success": True,
                "deleted": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "recursive": recursive,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to delete directory: {str(e)}"}
    
    def copy_directory(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a directory and its contents.
        
        Args:
            source: Source directory path
            destination: Destination directory path
            
        Returns:
            Dict with copy result
        """
        try:
            src_resolved = self.validator.resolve_path(source)
            dst_resolved = self.validator.resolve_path(destination)
            
            if not src_resolved.exists():
                return {"error": f"Source directory not found: {source}"}
            
            if not src_resolved.is_dir():
                return {"error": f"Source is not a directory: {source}"}
            
            if dst_resolved.exists():
                return {"error": f"Destination already exists: {destination}"}
            
            shutil.copytree(src_resolved, dst_resolved)
            
            return {
                "success": True,
                "source": str(src_resolved),
                "destination": str(dst_resolved),
                "relative_destination": self.validator.get_relative_path(dst_resolved),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to copy directory: {str(e)}"}
    
    def move_directory(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Move/rename a directory.
        
        Args:
            source: Source directory path
            destination: Destination directory path
            
        Returns:
            Dict with move result
        """
        try:
            src_resolved = self.validator.resolve_path(source)
            dst_resolved = self.validator.resolve_path(destination)
            
            if not src_resolved.exists():
                return {"error": f"Source directory not found: {source}"}
            
            if not src_resolved.is_dir():
                return {"error": f"Source is not a directory: {source}"}
            
            shutil.move(str(src_resolved), str(dst_resolved))
            
            return {
                "success": True,
                "source": str(src_resolved),
                "destination": str(dst_resolved),
                "relative_destination": self.validator.get_relative_path(dst_resolved),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to move directory: {str(e)}"}
    
    def get_directory_info(self, path: str = ".") -> Dict[str, Any]:
        """
        Get detailed information about a directory.
        
        Args:
            path: Directory path
            
        Returns:
            Dict with directory metadata and statistics
        """
        try:
            resolved = self.validator.resolve_path(path)
            
            if not resolved.exists():
                return {"error": f"Directory not found: {path}"}
            
            if not resolved.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            stat = resolved.stat()
            
            # Calculate stats
            total_files = 0
            total_folders = 0
            total_size = 0
            file_extensions = {}
            
            for item in resolved.rglob("*"):
                if item.is_file():
                    total_files += 1
                    total_size += item.stat().st_size
                    ext = item.suffix or "(no extension)"
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
                elif item.is_dir():
                    total_folders += 1
            
            return {
                "path": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "name": resolved.name,
                "total_files": total_files,
                "total_folders": total_folders,
                "total_size_bytes": total_size,
                "total_size_human": self._format_size(total_size),
                "file_extensions": dict(sorted(
                    file_extensions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to get directory info: {str(e)}"}
    
    def find_files(
        self,
        pattern: str,
        path: str = ".",
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find files matching a pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "*test*")
            path: Starting directory
            file_type: Filter by extension (e.g., ".py")
            
        Returns:
            Dict with matching files
        """
        try:
            resolved = self.validator.resolve_path(path)
            
            if not resolved.exists():
                return {"error": f"Directory not found: {path}"}
            
            matches = []
            
            for file_path in resolved.rglob(pattern):
                # Skip ignored
                if any(p in str(file_path) for p in self.settings.ignore_patterns):
                    continue
                
                # Filter by type
                if file_type and file_path.suffix != file_type:
                    continue
                
                if file_path.is_file():
                    matches.append({
                        "name": file_path.name,
                        "path": self.validator.get_relative_path(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "extension": file_path.suffix,
                    })
            
            return {
                "pattern": pattern,
                "search_path": str(resolved),
                "match_count": len(matches),
                "matches": matches[:100],  # Limit results
                "truncated": len(matches) > 100,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to find files: {str(e)}"}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"


# Tool function wrappers for agent
def list_directory(path: str = ".", recursive: bool = False) -> str:
    """List directory contents"""
    tools = FolderTools()
    result = tools.list_directory(path, recursive)
    if "error" in result:
        return f"Error: {result['error']}"
    
    if recursive:
        return result.get("tree_text", "")
    else:
        output = []
        for folder in result.get("folders", []):
            output.append(f"📁 {folder['name']}/")
        for file in result.get("files", []):
            output.append(f"📄 {file['name']}")
        return "\n".join(output) if output else "Directory is empty"


def create_directory(path: str) -> str:
    """Create a new directory"""
    tools = FolderTools()
    result = tools.create_directory(path)
    if "error" in result:
        return f"Error: {result['error']}"
    if result.get("already_exists"):
        return f"Directory already exists: {result['relative_path']}"
    return f"Created directory: {result['relative_path']}"


def delete_directory(path: str, recursive: bool = False) -> str:
    """Delete a directory"""
    tools = FolderTools()
    result = tools.delete_directory(path, recursive)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Deleted directory: {result['relative_path']}"


def find_files(pattern: str, path: str = ".") -> str:
    """Find files matching a pattern"""
    tools = FolderTools()
    result = tools.find_files(pattern, path)
    if "error" in result:
        return f"Error: {result['error']}"
    
    matches = result.get("matches", [])
    if not matches:
        return f"No files matching '{pattern}' found"
    
    output = [f"Found {result['match_count']} matches:"]
    for m in matches[:20]:
        output.append(f"  📄 {m['path']}")
    if len(matches) > 20:
        output.append(f"  ... and {len(matches) - 20} more")
    return "\n".join(output)



