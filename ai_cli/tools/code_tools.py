"""
Code Tools - Smart code editing with line-level precision
Provides intelligent code modification capabilities
"""

import re
import difflib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path, get_cache_path
from utils.security import PathValidator, SecurityError
from utils.parsers import PythonASTParser, get_file_structure


class CodeTools:
    """
    Provides intelligent code editing capabilities.
    Supports line-level operations, find/replace, and structure analysis.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.validator = PathValidator(self.sandbox_path)
        self.settings = get_settings()
        self.backup_dir = get_cache_path() / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PythonASTParser()
    
    def insert_lines(
        self,
        filepath: str,
        line_number: int,
        content: str,
    ) -> Dict[str, Any]:
        """
        Insert content at a specific line number.
        
        Args:
            filepath: Path to file
            line_number: Line number to insert at (1-indexed)
            content: Content to insert
            
        Returns:
            Dict with insertion result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            # Read existing content
            existing = resolved.read_text(encoding="utf-8")
            lines = existing.split("\n")
            
            # Validate line number
            if line_number < 1:
                line_number = 1
            if line_number > len(lines) + 1:
                line_number = len(lines) + 1
            
            # Create backup
            self._create_backup(resolved)
            
            # Insert content
            new_lines = content.split("\n")
            insert_index = line_number - 1
            
            result_lines = lines[:insert_index] + new_lines + lines[insert_index:]
            new_content = "\n".join(result_lines)
            
            resolved.write_text(new_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "inserted_at_line": line_number,
                "lines_inserted": len(new_lines),
                "new_total_lines": len(result_lines),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to insert lines: {str(e)}"}
    
    def replace_lines(
        self,
        filepath: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> Dict[str, Any]:
        """
        Replace a range of lines with new content.
        
        Args:
            filepath: Path to file
            start_line: Start line number (1-indexed, inclusive)
            end_line: End line number (1-indexed, inclusive)
            content: Replacement content
            
        Returns:
            Dict with replacement result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            # Read existing content
            existing = resolved.read_text(encoding="utf-8")
            lines = existing.split("\n")
            total_lines = len(lines)
            
            # Validate line numbers
            if start_line < 1:
                start_line = 1
            if end_line > total_lines:
                end_line = total_lines
            if start_line > end_line:
                return {"error": f"Invalid line range: {start_line}-{end_line}"}
            
            # Create backup
            self._create_backup(resolved)
            
            # Replace lines
            new_lines = content.split("\n")
            start_index = start_line - 1
            end_index = end_line
            
            replaced_lines = lines[start_index:end_index]
            result_lines = lines[:start_index] + new_lines + lines[end_index:]
            new_content = "\n".join(result_lines)
            
            resolved.write_text(new_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "replaced_range": f"{start_line}-{end_line}",
                "lines_removed": len(replaced_lines),
                "lines_inserted": len(new_lines),
                "new_total_lines": len(result_lines),
                "old_content": "\n".join(replaced_lines),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to replace lines: {str(e)}"}
    
    def delete_lines(
        self,
        filepath: str,
        start_line: int,
        end_line: int,
    ) -> Dict[str, Any]:
        """
        Delete a range of lines.
        
        Args:
            filepath: Path to file
            start_line: Start line number (1-indexed, inclusive)
            end_line: End line number (1-indexed, inclusive)
            
        Returns:
            Dict with deletion result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            # Read existing content
            existing = resolved.read_text(encoding="utf-8")
            lines = existing.split("\n")
            total_lines = len(lines)
            
            # Validate line numbers
            if start_line < 1:
                start_line = 1
            if end_line > total_lines:
                end_line = total_lines
            if start_line > end_line:
                return {"error": f"Invalid line range: {start_line}-{end_line}"}
            
            # Create backup
            self._create_backup(resolved)
            
            # Delete lines
            start_index = start_line - 1
            end_index = end_line
            
            deleted_lines = lines[start_index:end_index]
            result_lines = lines[:start_index] + lines[end_index:]
            new_content = "\n".join(result_lines)
            
            resolved.write_text(new_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "deleted_range": f"{start_line}-{end_line}",
                "lines_deleted": len(deleted_lines),
                "new_total_lines": len(result_lines),
                "deleted_content": "\n".join(deleted_lines),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to delete lines: {str(e)}"}
    
    def find_and_replace(
        self,
        filepath: str,
        find: str,
        replace: str,
        regex: bool = False,
        case_sensitive: bool = True,
        max_replacements: int = -1,
    ) -> Dict[str, Any]:
        """
        Find and replace text in a file.
        
        Args:
            filepath: Path to file
            find: Text or pattern to find
            replace: Replacement text
            regex: If True, treat find as regex pattern
            case_sensitive: If True, case-sensitive matching
            max_replacements: Max replacements (-1 = all)
            
        Returns:
            Dict with replacement result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            # Read existing content
            existing = resolved.read_text(encoding="utf-8")
            
            # Create backup
            self._create_backup(resolved)
            
            # Perform replacement
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    pattern = re.compile(find, flags)
                except re.error as e:
                    return {"error": f"Invalid regex pattern: {e}"}
                
                if max_replacements < 0:
                    new_content, count = pattern.subn(replace, existing)
                else:
                    new_content, count = pattern.subn(replace, existing, count=max_replacements)
            else:
                if not case_sensitive:
                    # Case-insensitive string replacement
                    pattern = re.compile(re.escape(find), re.IGNORECASE)
                    if max_replacements < 0:
                        new_content, count = pattern.subn(replace, existing)
                    else:
                        new_content, count = pattern.subn(replace, existing, count=max_replacements)
                else:
                    count = existing.count(find)
                    if max_replacements >= 0:
                        count = min(count, max_replacements)
                    new_content = existing.replace(find, replace, max_replacements if max_replacements >= 0 else -1)
            
            if count == 0:
                return {
                    "success": True,
                    "filepath": str(resolved),
                    "relative_path": self.validator.get_relative_path(resolved),
                    "replacements": 0,
                    "message": "No matches found",
                }
            
            resolved.write_text(new_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "find": find,
                "replace": replace,
                "replacements": count,
                "regex": regex,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to find and replace: {str(e)}"}
    
    def get_diff(
        self,
        filepath: str,
        new_content: str,
        context_lines: int = 3,
    ) -> Dict[str, Any]:
        """
        Get diff between current file and new content.
        
        Args:
            filepath: Path to file
            new_content: New content to compare
            context_lines: Number of context lines around changes
            
        Returns:
            Dict with diff result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            existing = resolved.read_text(encoding="utf-8")
            
            # Generate diff
            diff = difflib.unified_diff(
                existing.split("\n"),
                new_content.split("\n"),
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
                lineterm="",
                n=context_lines,
            )
            
            diff_lines = list(diff)
            
            # Count changes
            additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
            
            return {
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "diff": "\n".join(diff_lines),
                "additions": additions,
                "deletions": deletions,
                "has_changes": len(diff_lines) > 0,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to generate diff: {str(e)}"}
    
    def analyze_code(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze code structure of a Python file.
        
        Args:
            filepath: Path to Python file
            
        Returns:
            Dict with code analysis
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            if resolved.suffix != ".py":
                return {"error": "Code analysis only supports Python files"}
            
            structure = get_file_structure(resolved)
            
            return {
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                **structure,
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to analyze code: {str(e)}"}
    
    def undo_last_edit(self, filepath: str) -> Dict[str, Any]:
        """
        Restore file from backup (undo last edit).
        
        Args:
            filepath: Path to file
            
        Returns:
            Dict with undo result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            backup_path = self._get_backup_path(resolved)
            
            if not backup_path.exists():
                return {"error": f"No backup found for: {filepath}"}
            
            # Restore from backup
            backup_content = backup_path.read_text(encoding="utf-8")
            resolved.write_text(backup_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "restored_from": str(backup_path),
                "message": "File restored from backup",
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to undo: {str(e)}"}
    
    def apply_edit(
        self,
        filepath: str,
        old_content: str,
        new_content: str,
    ) -> Dict[str, Any]:
        """
        Apply an edit by finding and replacing exact content.
        This is useful for precise code modifications.
        
        Args:
            filepath: Path to file
            old_content: Exact content to find
            new_content: Content to replace with
            
        Returns:
            Dict with edit result
        """
        try:
            resolved = self.validator.resolve_path(filepath)
            
            if not resolved.exists():
                return {"error": f"File not found: {filepath}"}
            
            existing = resolved.read_text(encoding="utf-8")
            
            # Check if old content exists
            if old_content not in existing:
                # Try to find similar content
                similar = self._find_similar(existing, old_content)
                return {
                    "error": "Old content not found in file",
                    "similar_matches": similar,
                }
            
            # Check for unique match
            count = existing.count(old_content)
            if count > 1:
                return {
                    "error": f"Found {count} matches. Please provide more specific content.",
                    "match_count": count,
                }
            
            # Create backup
            self._create_backup(resolved)
            
            # Apply edit
            new_file_content = existing.replace(old_content, new_content, 1)
            resolved.write_text(new_file_content, encoding="utf-8")
            
            return {
                "success": True,
                "filepath": str(resolved),
                "relative_path": self.validator.get_relative_path(resolved),
                "old_length": len(old_content),
                "new_length": len(new_content),
            }
            
        except SecurityError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to apply edit: {str(e)}"}
    
    def _create_backup(self, filepath: Path):
        """Create a backup of the file"""
        backup_path = self._get_backup_path(filepath)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, backup_path)
    
    def _get_backup_path(self, filepath: Path) -> Path:
        """Get backup path for a file"""
        relative = filepath.relative_to(self.sandbox_path)
        return self.backup_dir / f"{relative}.bak"
    
    def _find_similar(self, content: str, search: str, max_results: int = 3) -> List[str]:
        """Find similar content using fuzzy matching"""
        lines = content.split("\n")
        search_lines = search.split("\n")
        
        # Use difflib to find similar sequences
        matcher = difflib.SequenceMatcher()
        matcher.set_seq2(search)
        
        similar = []
        window_size = len(search_lines)
        
        for i in range(len(lines) - window_size + 1):
            window = "\n".join(lines[i:i + window_size])
            matcher.set_seq1(window)
            ratio = matcher.ratio()
            
            if ratio > 0.6:  # 60% similarity threshold
                similar.append({
                    "line": i + 1,
                    "content": window[:200] + "..." if len(window) > 200 else window,
                    "similarity": f"{ratio:.0%}",
                })
        
        # Sort by similarity and limit
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:max_results]


# Tool function wrappers for agent
def insert_lines(filepath: str, line_number: int, content: str) -> str:
    """Insert content at a specific line"""
    tools = CodeTools()
    result = tools.insert_lines(filepath, line_number, content)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Inserted {result['lines_inserted']} lines at line {result['inserted_at_line']}"


def replace_lines(filepath: str, start_line: int, end_line: int, content: str) -> str:
    """Replace a range of lines"""
    tools = CodeTools()
    result = tools.replace_lines(filepath, start_line, end_line, content)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Replaced lines {result['replaced_range']} ({result['lines_removed']} removed, {result['lines_inserted']} inserted)"


def delete_lines(filepath: str, start_line: int, end_line: int) -> str:
    """Delete a range of lines"""
    tools = CodeTools()
    result = tools.delete_lines(filepath, start_line, end_line)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Deleted lines {result['deleted_range']} ({result['lines_deleted']} lines)"


def find_and_replace(filepath: str, find: str, replace: str, regex: bool = False) -> str:
    """Find and replace text in a file"""
    tools = CodeTools()
    result = tools.find_and_replace(filepath, find, replace, regex)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Made {result['replacements']} replacements"


def analyze_code(filepath: str) -> str:
    """Analyze code structure"""
    import json
    tools = CodeTools()
    result = tools.analyze_code(filepath)
    if "error" in result:
        return f"Error: {result['error']}"
    return json.dumps(result, indent=2)



