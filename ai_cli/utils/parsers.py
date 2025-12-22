"""
Python code parsers for AST analysis and symbol extraction
Provides deep code understanding for intelligent indexing
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SymbolType(Enum):
    """Types of code symbols"""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    DECORATOR = "decorator"


@dataclass
class CodeSymbol:
    """Represents a code symbol with full metadata"""
    name: str
    symbol_type: SymbolType
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    signature: str = ""
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent class/function name
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)
    complexity: int = 1  # Cyclomatic complexity estimate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.symbol_type.value,
            "file": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent": self.parent,
            "decorators": self.decorators,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "base_classes": self.base_classes,
            "complexity": self.complexity,
        }


@dataclass
class ImportInfo:
    """Represents an import statement"""
    module: str
    names: List[str]  # Imported names (or ["*"] for star import)
    alias: Optional[str] = None
    is_from_import: bool = False
    line_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "is_from": self.is_from_import,
            "line": self.line_number,
        }


@dataclass
class ParseResult:
    """Complete parsing result for a Python file"""
    file_path: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # __all__ if defined
    module_docstring: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    total_lines: int = 0
    code_lines: int = 0  # Non-empty, non-comment lines
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": [i.to_dict() for i in self.imports],
            "exports": self.exports,
            "module_docstring": self.module_docstring,
            "errors": self.errors,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
        }


class PythonASTParser:
    """
    Advanced Python AST parser for deep code analysis.
    Extracts symbols, relationships, and structural information.
    """
    
    def __init__(self):
        self.current_file: str = ""
        self.source_lines: List[str] = []
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a Python file and extract all symbols and metadata"""
        self.current_file = str(file_path)
        result = ParseResult(file_path=self.current_file)
        
        try:
            content = file_path.read_text(encoding="utf-8")
            self.source_lines = content.split("\n")
            result.total_lines = len(self.source_lines)
            result.code_lines = self._count_code_lines(content)
            
            tree = ast.parse(content, filename=self.current_file)
            
            # Extract module docstring
            result.module_docstring = ast.get_docstring(tree)
            
            # Extract __all__ exports
            result.exports = self._extract_exports(tree)
            
            # Extract imports
            result.imports = self._extract_imports(tree)
            
            # Extract all symbols
            result.symbols = self._extract_symbols(tree)
            
        except SyntaxError as e:
            result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")
        
        return result
    
    def parse_code(self, code: str, filename: str = "<string>") -> ParseResult:
        """Parse code string directly"""
        self.current_file = filename
        self.source_lines = code.split("\n")
        result = ParseResult(file_path=filename)
        result.total_lines = len(self.source_lines)
        result.code_lines = self._count_code_lines(code)
        
        try:
            tree = ast.parse(code, filename=filename)
            result.module_docstring = ast.get_docstring(tree)
            result.exports = self._extract_exports(tree)
            result.imports = self._extract_imports(tree)
            result.symbols = self._extract_symbols(tree)
        except SyntaxError as e:
            result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")
        
        return result
    
    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines"""
        count = 0
        in_multiline_string = False
        
        for line in content.split("\n"):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Handle multiline strings (simplified)
            if '"""' in stripped or "'''" in stripped:
                count += 1
                continue
            
            # Skip single-line comments
            if stripped.startswith("#"):
                continue
            
            count += 1
        
        return count
    
    def _extract_exports(self, tree: ast.AST) -> List[str]:
        """Extract __all__ definition if present"""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            return [
                                elt.value for elt in node.value.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            ]
        return []
    
    def _extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract all import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name.split(".")[-1]],
                        alias=alias.asname,
                        is_from_import=False,
                        line_number=node.lineno,
                    ))
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    alias=node.names[0].asname if len(node.names) == 1 else None,
                    is_from_import=True,
                    line_number=node.lineno,
                ))
        
        return imports
    
    def _extract_symbols(self, tree: ast.AST, parent: Optional[str] = None) -> List[CodeSymbol]:
        """Recursively extract all symbols from AST"""
        symbols = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(self._parse_class(node, parent))
                # Recursively get methods
                symbols.extend(self._extract_symbols(node, parent=node.name))
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._parse_function(node, parent))
                # Don't recurse into nested functions for top-level symbols
            
            elif isinstance(node, ast.Assign) and parent is None:
                # Module-level assignments (variables/constants)
                symbols.extend(self._parse_assignment(node))
        
        return symbols
    
    def _parse_class(self, node: ast.ClassDef, parent: Optional[str]) -> CodeSymbol:
        """Parse a class definition"""
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Build signature
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        signature += ":"
        
        return CodeSymbol(
            name=node.name,
            symbol_type=SymbolType.CLASS,
            file_path=self.current_file,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            signature=signature,
            docstring=ast.get_docstring(node),
            parent=parent,
            decorators=decorators,
            base_classes=bases,
            complexity=self._estimate_complexity(node),
        )
    
    def _parse_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: Optional[str]) -> CodeSymbol:
        """Parse a function or method definition"""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        is_method = parent is not None
        
        # Get parameters
        params = self._get_function_params(node)
        
        # Get return type annotation
        return_type = None
        if node.returns:
            return_type = self._get_annotation_str(node.returns)
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Build signature
        prefix = "async def" if is_async else "def"
        params_str = ", ".join(params)
        signature = f"{prefix} {node.name}({params_str})"
        if return_type:
            signature += f" -> {return_type}"
        signature += ":"
        
        # Determine symbol type
        if is_async:
            symbol_type = SymbolType.ASYNC_FUNCTION
        elif is_method:
            symbol_type = SymbolType.METHOD
        else:
            symbol_type = SymbolType.FUNCTION
        
        return CodeSymbol(
            name=node.name,
            symbol_type=symbol_type,
            file_path=self.current_file,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            signature=signature,
            docstring=ast.get_docstring(node),
            parent=parent,
            decorators=decorators,
            parameters=params,
            return_type=return_type,
            complexity=self._estimate_complexity(node),
        )
    
    def _parse_assignment(self, node: ast.Assign) -> List[CodeSymbol]:
        """Parse module-level variable assignments"""
        symbols = []
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                # Determine if it's a constant (UPPER_CASE)
                is_constant = name.isupper() or name.startswith("_") and name[1:].isupper()
                
                # Try to get value representation
                value_repr = self._get_value_repr(node.value)
                signature = f"{name} = {value_repr}"
                
                symbols.append(CodeSymbol(
                    name=name,
                    symbol_type=SymbolType.CONSTANT if is_constant else SymbolType.VARIABLE,
                    file_path=self.current_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    column_start=node.col_offset,
                    signature=signature,
                ))
        
        return symbols
    
    def _get_function_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract function parameters with type annotations"""
        params = []
        args = node.args
        
        # Regular arguments
        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param = arg.arg
            if arg.annotation:
                param += f": {self._get_annotation_str(arg.annotation)}"
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                default_val = self._get_value_repr(args.defaults[default_idx])
                param += f" = {default_val}"
            params.append(param)
        
        # *args
        if args.vararg:
            param = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                param += f": {self._get_annotation_str(args.vararg.annotation)}"
            params.append(param)
        
        # **kwargs
        if args.kwarg:
            param = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                param += f": {self._get_annotation_str(args.kwarg.annotation)}"
            params.append(param)
        
        return params
    
    def _get_annotation_str(self, node: ast.AST) -> str:
        """Convert annotation AST node to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_str(node.value)
            slice_str = self._get_annotation_str(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Tuple):
            items = [self._get_annotation_str(elt) for elt in node.elts]
            return ", ".join(items)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type (3.10+ syntax: X | Y)
            left = self._get_annotation_str(node.left)
            right = self._get_annotation_str(node.right)
            return f"{left} | {right}"
        return "..."
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., module.Class)"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "unknown"
    
    def _get_value_repr(self, node: ast.AST, max_length: int = 50) -> str:
        """Get string representation of a value"""
        if isinstance(node, ast.Constant):
            value = repr(node.value)
            if len(value) > max_length:
                return value[:max_length] + "..."
            return value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[...]"
        elif isinstance(node, ast.Dict):
            return "{...}"
        elif isinstance(node, ast.Set):
            return "{...}"
        elif isinstance(node, ast.Tuple):
            return "(...)"
        elif isinstance(node, ast.Call):
            func_name = self._get_decorator_name(node.func)
            return f"{func_name}(...)"
        return "..."
    
    def _estimate_complexity(self, node: ast.AST) -> int:
        """
        Estimate cyclomatic complexity.
        Counts decision points: if, for, while, except, and, or, comprehensions
        """
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity


class CodeChunker:
    """
    Splits code into semantic chunks for embedding.
    Preserves logical boundaries (functions, classes).
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.parser = PythonASTParser()
    
    def chunk_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Split a Python file into semantic chunks.
        Each chunk includes metadata about its context.
        """
        chunks = []
        
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
        except Exception as e:
            return [{
                "content": f"Error reading file: {e}",
                "file": str(file_path),
                "start_line": 0,
                "end_line": 0,
                "type": "error",
            }]
        
        # Parse to get symbol boundaries
        parse_result = self.parser.parse_file(file_path)
        
        if parse_result.errors:
            # Fall back to simple line-based chunking
            return self._simple_chunk(content, str(file_path))
        
        # Get all symbol boundaries
        boundaries = [(s.line_start - 1, s.line_end - 1, s) for s in parse_result.symbols]
        boundaries.sort(key=lambda x: x[0])
        
        # Create chunks based on symbols
        current_pos = 0
        
        for start, end, symbol in boundaries:
            # Add any content before this symbol
            if start > current_pos:
                pre_content = "\n".join(lines[current_pos:start])
                if pre_content.strip():
                    chunks.append({
                        "content": pre_content,
                        "file": str(file_path),
                        "start_line": current_pos + 1,
                        "end_line": start,
                        "type": "module_code",
                        "context": None,
                    })
            
            # Add the symbol itself
            symbol_content = "\n".join(lines[start:end + 1])
            
            # If symbol is too large, split it
            if len(symbol_content) > self.chunk_size * 1.5:
                sub_chunks = self._split_large_symbol(symbol_content, symbol, start)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": symbol_content,
                    "file": str(file_path),
                    "start_line": start + 1,
                    "end_line": end + 1,
                    "type": symbol.symbol_type.value,
                    "context": symbol.to_dict(),
                })
            
            current_pos = end + 1
        
        # Add remaining content
        if current_pos < len(lines):
            remaining = "\n".join(lines[current_pos:])
            if remaining.strip():
                chunks.append({
                    "content": remaining,
                    "file": str(file_path),
                    "start_line": current_pos + 1,
                    "end_line": len(lines),
                    "type": "module_code",
                    "context": None,
                })
        
        return chunks
    
    def _split_large_symbol(self, content: str, symbol: CodeSymbol, base_line: int) -> List[Dict[str, Any]]:
        """Split a large symbol into smaller chunks with context"""
        chunks = []
        lines = content.split("\n")
        
        # Include signature/header in each chunk
        header_lines = min(3, len(lines))
        header = "\n".join(lines[:header_lines])
        
        # Chunk the body
        body_start = header_lines
        while body_start < len(lines):
            body_end = min(body_start + (self.chunk_size // 50), len(lines))  # ~50 chars per line estimate
            
            chunk_content = header + "\n# ... (continued)\n" + "\n".join(lines[body_start:body_end])
            
            chunks.append({
                "content": chunk_content,
                "file": symbol.file_path,
                "start_line": base_line + body_start + 1,
                "end_line": base_line + body_end,
                "type": symbol.symbol_type.value,
                "context": symbol.to_dict(),
                "is_partial": True,
            })
            
            body_start = body_end - (self.overlap // 50)  # Overlap
        
        return chunks
    
    def _simple_chunk(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Simple line-based chunking for non-Python or unparseable files"""
        chunks = []
        lines = content.split("\n")
        
        chunk_lines = self.chunk_size // 50  # Estimate
        overlap_lines = self.overlap // 50
        
        start = 0
        while start < len(lines):
            end = min(start + chunk_lines, len(lines))
            chunk_content = "\n".join(lines[start:end])
            
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


# Convenience functions
def parse_python_file(file_path: Path) -> ParseResult:
    """Parse a Python file and return results"""
    parser = PythonASTParser()
    return parser.parse_file(file_path)


def extract_symbols(file_path: Path) -> List[CodeSymbol]:
    """Extract all symbols from a Python file"""
    result = parse_python_file(file_path)
    return result.symbols


def get_file_structure(file_path: Path) -> Dict[str, Any]:
    """Get a structured overview of a Python file"""
    result = parse_python_file(file_path)
    
    return {
        "file": str(file_path),
        "lines": result.total_lines,
        "code_lines": result.code_lines,
        "docstring": result.module_docstring,
        "imports": len(result.imports),
        "classes": [s.name for s in result.symbols if s.symbol_type == SymbolType.CLASS],
        "functions": [s.name for s in result.symbols if s.symbol_type in (SymbolType.FUNCTION, SymbolType.ASYNC_FUNCTION)],
        "exports": result.exports,
        "errors": result.errors,
    }



