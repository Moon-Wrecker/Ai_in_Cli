#!/usr/bin/env python3
"""
Smart File Editor with RAG integration
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class EditContext:
    """Context for file editing operations"""
    file_path: str
    content: str
    line_count: int
    language: Optional[str] = None
    encoding: str = "utf-8"

@dataclass
class EditOperation:
    """Represents a file edit operation"""
    operation_type: str  # 'insert', 'replace', 'delete', 'append'
    line_number: Optional[int] = None
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    search_pattern: Optional[str] = None

class SmartFileEditor:
    """Advanced file editor with context awareness"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else None
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml'
        }
    
    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path within base directory"""
        path = Path(file_path)
        if self.base_dir:
            if not path.is_absolute():
                path = self.base_dir / path
            path = path.resolve()
            if not path.is_relative_to(self.base_dir):
                raise ValueError(f"Path {path} is outside allowed directory")
        return path
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.language_extensions.get(ext, 'text')
    
    def _get_file_context(self, file_path: str) -> EditContext:
        """Get file context for editing"""
        path = self._resolve_path(file_path)
        
        if not path.exists():
            return EditContext(
                file_path=str(path),
                content="",
                line_count=0,
                language=self._detect_language(str(path))
            )
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        return EditContext(
            file_path=str(path),
            content=content,
            line_count=len(content.splitlines()),
            language=self._detect_language(str(path))
        )
    
    def read_file_with_lines(self, file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Read file with line numbers"""
        try:
            context = self._get_file_context(file_path)
            lines = context.content.splitlines()
            
            if end_line is None:
                end_line = len(lines)
            
            # Ensure valid line numbers
            start_line = max(1, min(start_line, len(lines)))
            end_line = max(start_line, min(end_line, len(lines)))
            
            selected_lines = lines[start_line-1:end_line]
            
            result = {
                "success": True,
                "file_path": file_path,
                "total_lines": len(lines),
                "start_line": start_line,
                "end_line": end_line,
                "language": context.language,
                "content": "\n".join(selected_lines),
                "lines_with_numbers": []
            }
            
            # Add line numbers
            for i, line in enumerate(selected_lines, start=start_line):
                result["lines_with_numbers"].append(f"{i:4d}: {line}")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def insert_lines(self, file_path: str, line_number: int, content: str) -> Dict[str, Any]:
        """Insert content at specific line number"""
        try:
            context = self._get_file_context(file_path)
            lines = context.content.splitlines()
            
            # Insert at specified line (1-based indexing)
            insert_index = max(0, min(line_number - 1, len(lines)))
            new_lines = content.splitlines()
            
            # Insert the new lines
            result_lines = lines[:insert_index] + new_lines + lines[insert_index:]
            
            # Write back to file
            path = self._resolve_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(result_lines))
            
            return {
                "success": True,
                "operation": "insert",
                "file_path": file_path,
                "line_number": line_number,
                "lines_inserted": len(new_lines),
                "total_lines": len(result_lines),
                "message": f"Inserted {len(new_lines)} lines at line {line_number}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "insert",
                "file_path": file_path
            }
    
    def replace_lines(self, file_path: str, start_line: int, end_line: int, content: str) -> Dict[str, Any]:
        """Replace lines in a specific range"""
        try:
            context = self._get_file_context(file_path)
            lines = context.content.splitlines()
            
            # Ensure valid line numbers (1-based indexing)
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            new_lines = content.splitlines()
            
            # Replace the lines
            result_lines = lines[:start_idx] + new_lines + lines[end_idx:]
            
            # Write back to file
            path = self._resolve_path(file_path)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(result_lines))
            
            return {
                "success": True,
                "operation": "replace",
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_replaced": end_idx - start_idx,
                "lines_inserted": len(new_lines),
                "total_lines": len(result_lines),
                "message": f"Replaced lines {start_line}-{end_line} with {len(new_lines)} lines"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "replace",
                "file_path": file_path
            }
    
    def find_and_replace(self, file_path: str, search_pattern: str, replacement: str, is_regex: bool = False) -> Dict[str, Any]:
        """Find and replace text in file"""
        try:
            context = self._get_file_context(file_path)
            
            if is_regex:
                new_content, count = re.subn(search_pattern, replacement, context.content)
            else:
                new_content = context.content.replace(search_pattern, replacement)
                count = context.content.count(search_pattern)
            
            # Write back to file
            path = self._resolve_path(file_path)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "operation": "find_replace",
                "file_path": file_path,
                "search_pattern": search_pattern,
                "replacement": replacement,
                "is_regex": is_regex,
                "replacements_made": count,
                "message": f"Made {count} replacements"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "find_replace",
                "file_path": file_path
            }
    
    def delete_lines(self, file_path: str, start_line: int, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Delete specified lines from file"""
        try:
            if end_line is None:
                end_line = start_line
                
            context = self._get_file_context(file_path)
            lines = context.content.splitlines()
            
            # Ensure valid line numbers (1-based indexing)
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            # Delete the lines
            result_lines = lines[:start_idx] + lines[end_idx:]
            
            # Write back to file
            path = self._resolve_path(file_path)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(result_lines))
            
            return {
                "success": True,
                "operation": "delete",
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_deleted": end_idx - start_idx,
                "total_lines": len(result_lines),
                "message": f"Deleted lines {start_line}-{end_line}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "delete",
                "file_path": file_path
            }
    
    def append_to_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Append content to end of file"""
        try:
            path = self._resolve_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append content
            with open(path, 'a', encoding='utf-8') as f:
                if not content.endswith('\n'):
                    content += '\n'
                f.write(content)
            
            # Get updated line count
            with open(path, 'r', encoding='utf-8') as f:
                total_lines = len(f.readlines())
            
            return {
                "success": True,
                "operation": "append",
                "file_path": file_path,
                "content_length": len(content),
                "total_lines": total_lines,
                "message": f"Appended content to {file_path}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "append",
                "file_path": file_path
            }
    
    def analyze_file_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze file structure (functions, classes, etc.)"""
        try:
            context = self._get_file_context(file_path)
            
            analysis = {
                "success": True,
                "file_path": file_path,
                "language": context.language,
                "total_lines": context.line_count,
                "functions": [],
                "classes": [],
                "imports": [],
                "structure": []
            }
            
            if context.language == 'python':
                analysis.update(self._analyze_python_file(context.content))
            elif context.language in ['javascript', 'typescript']:
                analysis.update(self._analyze_js_file(context.content))
            
            return analysis
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def _analyze_python_file(self, content: str) -> Dict[str, List]:
        """Analyze Python file structure"""
        lines = content.splitlines()
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "structure": []
        }
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Imports
            if stripped.startswith('import ') or stripped.startswith('from '):
                analysis["imports"].append({
                    "line": i,
                    "statement": stripped
                })
            
            # Classes
            if stripped.startswith('class '):
                class_match = re.match(r'class\s+(\w+)', stripped)
                if class_match:
                    analysis["classes"].append({
                        "line": i,
                        "name": class_match.group(1),
                        "definition": stripped
                    })
                    analysis["structure"].append(f"Line {i}: Class {class_match.group(1)}")
            
            # Functions
            if stripped.startswith('def '):
                func_match = re.match(r'def\s+(\w+)', stripped)
                if func_match:
                    analysis["functions"].append({
                        "line": i,
                        "name": func_match.group(1),
                        "definition": stripped
                    })
                    analysis["structure"].append(f"Line {i}: Function {func_match.group(1)}")
        
        return analysis
    
    def _analyze_js_file(self, content: str) -> Dict[str, List]:
        """Analyze JavaScript/TypeScript file structure"""
        lines = content.splitlines()
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "structure": []
        }
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Imports
            if stripped.startswith('import ') or stripped.startswith('const ') and 'require(' in stripped:
                analysis["imports"].append({
                    "line": i,
                    "statement": stripped
                })
            
            # Classes
            if stripped.startswith('class '):
                class_match = re.match(r'class\s+(\w+)', stripped)
                if class_match:
                    analysis["classes"].append({
                        "line": i,
                        "name": class_match.group(1),
                        "definition": stripped
                    })
                    analysis["structure"].append(f"Line {i}: Class {class_match.group(1)}")
            
            # Functions
            if 'function ' in stripped or '=>' in stripped:
                if stripped.startswith('function '):
                    func_match = re.match(r'function\s+(\w+)', stripped)
                    if func_match:
                        analysis["functions"].append({
                            "line": i,
                            "name": func_match.group(1),
                            "definition": stripped
                        })
                        analysis["structure"].append(f"Line {i}: Function {func_match.group(1)}")
        
        return analysis
