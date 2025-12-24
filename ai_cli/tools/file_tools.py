"""
File Tools - CRUD operations for files
All operations are sandboxed to the sandbox directory
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from utils.security import PathValidator, SecurityError, get_path_validator


class FileTools:
    """
    Provides file CRUD operations with security sandboxing.
    All operations are restricted to the sandbox directory.
    Automatically triggers incremental indexing for Python files.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None, auto_index: bool = True):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.validator = PathValidator(self.sandbox_path)
        self.settings = get_settings()
        self.auto_index = auto_index  # Enable/disable auto-indexing
        self._index_manager = None
    
    @property
    def index_manager(self):
        """Lazy load index manager"""
        if self._index_manager is None and self.auto_index:
            try:
                from indexing.index_manager import get_index_manager
                self._index_manager = get_index_manager(self.sandbox_path)
            except Exception:
                pass  # Indexing not available
        return self._index_manager
    
    def _trigger_index(self, file_path: Path, action: str):
        """Trigger indexing for a file after an operation"""
        if not self.auto_index or not self.index_manager:
            return
        
        # Only index Python files
        if file_path.suffix != ".py":
            return
        
        try:
            if action == "deleted":
                self.index_manager.remove_file_from_index(file_path)
            else:
                self.index_manager.index_file(file_path, reason=action)
        except Exception:
            pass  # Don't fail the operation if indexing fails
    
    def read_file(self, filepath: str) -> Dict[str, Any]:
        """
        Read the contents of a file.
        
        Args:
            filepath: Path to file (relative to sandbox or absolute within sandbox)
            
        Returns:
            Dict with file content, line count, and metadata
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            if not resolved.is_file():
                return {"error": f"Not a file: {filepath}"}
            
            # Check file size
            size = resolved.stat().st_size
            size_mb = size / (1024 * 1024)
            
            if size_mb > self.settings.max_file_size_mb:
                return {
                    "error": f"File too large ({size_mb:.2f}MB). Max: {self.settings.max_file_size_mb}MB",
                    "size_bytes": size,
                }
            
            # Try to read with different encodings
            content = None
            encoding_used = None
            
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    content = resolved.read_text(encoding=encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # Fall back to binary
                binary_content = resolved.read_bytes()
                return {
                    "filepath": str(resolved),
                    "relative_path": self.validator.get_relative_path(resolved),
                    "is_binary": True,
                    "size_bytes": size,
                    "content": f"<binary file: {size} bytes>",
                }
            
            lines = content.split("\n")
            
            return {
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "content": content,
                "line_count": len(lines),
                "size_bytes": size,
                "encoding": encoding_used,
                "is_binary": False,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    def read_file_lines(
        self,
        filepath: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read specific lines from a file with line numbers.
        
        Args:
            filepath: Path to file
            start_line: Starting line (1-indexed, inclusive)
            end_line: Ending line (1-indexed, inclusive). None = end of file.
            
        Returns:
            Dict with numbered lines and metadata
        """
        result = self.read_file(filepath)
        
        if "error" in result:
            return result
        
        if result.get("is_binary"):
            return {"error": "Cannot read lines from binary file"}
        
        lines = result["content"].split("\n")
        total_lines = len(lines)
        
        # Validate line numbers
        start_line = max(1, start_line)
        if end_line is None:
            end_line = total_lines
        end_line = min(end_line, total_lines)
        
        if start_line > total_lines:
            return {
                "error": f"Start line {start_line} exceeds file length ({total_lines} lines)",
                "total_lines": total_lines,
            }
        
        # Extract lines with numbers
        selected_lines = []
        for i in range(start_line - 1, end_line):
            selected_lines.append({
                "number": i + 1,
                "content": lines[i],
            })
        
        return {
            "filepath": result["filepath"],
            "relative_path": result["relative_path"],
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "lines": selected_lines,
            "content_with_numbers": "\n".join(
                f"{line['number']:>6}| {line['content']}" for line in selected_lines
            ),
        }
    
    def create_file(
        self,
        filepath: str,
        content: str = "",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new file with content.
        
        Args:
            filepath: Path for new file
            content: File content
            overwrite: If True, overwrite existing file
            
        Returns:
            Dict with creation result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            # Check if file exists
            if resolved.exists() and not overwrite:
                return {
                    "error": f"File already exists: {filepath}. Use overwrite=True to replace.",
                    "filepath": str(resolved),
                }
            
            # Create parent directories if needed
            resolved.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            was_existing = resolved.exists()
            resolved.write_text(content, encoding="utf-8")
            
            # Auto-index if Python file
            self._trigger_index(resolved, "created" if not was_existing else "modified")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "size_bytes": len(content.encode("utf-8")),
                "line_count": len(content.split("\n")),
                "created": not was_existing,
                "overwritten": was_existing and overwrite,
                "indexed": resolved.suffix == ".py",
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to create file: {str(e)}"}
    
    def write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """
        Write/overwrite a file with content.
        
        Args:
            filepath: Path to file
            content: New content
            
        Returns:
            Dict with write result
        """
        return self.create_file(filepath, content, overwrite=True)
    
    def append_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """
        Append content to an existing file.
        
        Args:
            filepath: Path to file
            content: Content to append
            
        Returns:
            Dict with append result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            # Read existing content
            existing = resolved.read_text(encoding="utf-8")
            
            # Append new content
            new_content = existing + content
            resolved.write_text(new_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "appended_bytes": len(content.encode("utf-8")),
                "total_size_bytes": len(new_content.encode("utf-8")),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to append to file: {str(e)}"}
    
    def delete_file(self, filepath: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dict with deletion result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            if not resolved.is_file():
                return {"error": f"Not a file: {filepath}. Use delete_folder for directories."}
            
            # Store file info before deletion
            size = resolved.stat().st_size
            was_python = resolved.suffix == ".py"
            
            resolved.unlink()
            
            # Remove from index if Python file
            if was_python:
                self._trigger_index(resolved, "deleted")
            
            return {
                "success": True,
                "deleted": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "size_bytes": size,
                "removed_from_index": was_python,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to delete file: {str(e)}"}
    
    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a file to a new location.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            Dict with copy result
        """
        try:
            src_resolved = self.validator.resolve_path(source)
            dst_resolved = self.validator.resolve_path(destination)
            
            if not src_resolved.exists():
                return {"error": f"Source file not found: {source}"}
            
            if not src_resolved.is_file():
                return {"error": f"Source is not a file: {source}"}
            
            # Create destination directory if needed
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_resolved, dst_resolved)
            
            return {
                "success": True,
                "source": str(src_resolved),
                "destination": str(dst_resolved),
                "relative_destination": self.validator.get_relative_path(dst_resolved),
                "size_bytes": dst_resolved.stat().st_size,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to copy file: {str(e)}"}
    
    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Move/rename a file.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            Dict with move result
        """
        try:
            src_resolved = self.validator.resolve_path(source)
            dst_resolved = self.validator.resolve_path(destination)
            
            if not src_resolved.exists():
                return {"error": f"Source file not found: {source}"}
            
            if not src_resolved.is_file():
                return {"error": f"Source is not a file: {source}"}
            
            # Create destination directory if needed
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            
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
            return {"error": f"Failed to move file: {str(e)}"}
    
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get detailed information about a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dict with file metadata
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            stat = resolved.stat()
            
            return {
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "name": resolved.name,
                "extension": resolved.suffix,
                "is_file": resolved.is_file(),
                "is_directory": resolved.is_dir(),
                "size_bytes": stat.st_size,
                "size_human": self._format_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to get file info: {str(e)}"}
    
    def search_in_file(
        self,
        filepath: str,
        pattern: str,
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for a pattern in a file.
        
        Args:
            filepath: Path to file
            pattern: Search pattern (string or regex)
            case_sensitive: Whether search is case sensitive
            
        Returns:
            Dict with matching lines
        """
        import re
        
        result = self.read_file(filepath)
        
        if "error" in result:
            return result
        
        if result.get("is_binary"):
            return {"error": "Cannot search in binary file"}
        
        lines = result["content"].split("\n")
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}
        
        matches = []
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append({
                    "line_number": i,
                    "content": line,
                })
        
        return {
            "filepath": result["filepath"],
            "relative_path": result["relative_path"],
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches,
        }
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"


# Tool function wrappers for agent
def read_file(filepath: str) -> str:
    """Read file contents"""
    tools = FileTools()
    result = tools.read_file(filepath)
    if "error" in result:
        return f"Error: {result['error']}"
    return result.get("content", "")


def read_file_lines(filepath: str, start_line: int = 1, end_line: int = None) -> str:
    """Read specific lines from a file"""
    tools = FileTools()
    result = tools.read_file_lines(filepath, start_line, end_line)
    if "error" in result:
        return f"Error: {result['error']}"
    return result.get("content_with_numbers", "")


def create_file(filepath: str, content: str = "") -> str:
    """Create a new file"""
    tools = FileTools()
    result = tools.create_file(filepath, content)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Created file: {result['relative_path']}"


def write_file(filepath: str, content: str) -> str:
    """Write content to a file"""
    tools = FileTools()
    result = tools.write_file(filepath, content)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Wrote to file: {result['relative_path']}"


def delete_file(filepath: str) -> str:
    """Delete a file"""
    tools = FileTools()
    result = tools.delete_file(filepath)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Deleted file: {result['relative_path']}"



