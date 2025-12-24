"""
Multi-language code parsers for AST analysis and symbol extraction
Supports Python (AST), JavaScript/TypeScript, HTML, CSS parsing
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
    # JavaScript/TypeScript specific
    ARROW_FUNCTION = "arrow_function"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    # HTML specific
    HTML_ELEMENT = "html_element"
    HTML_COMPONENT = "html_component"
    # CSS specific
    CSS_SELECTOR = "css_selector"
    CSS_CLASS = "css_class"
    CSS_ID = "css_id"
    CSS_VARIABLE = "css_variable"
    CSS_KEYFRAMES = "css_keyframes"
    CSS_MEDIA = "css_media"


# Supported languages
SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
}


def get_language(file_path: Path) -> str:
    """Detect language from file extension"""
    return SUPPORTED_LANGUAGES.get(file_path.suffix.lower(), "unknown")


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


# =============================================================================
# JavaScript/TypeScript Parser
# =============================================================================

class JavaScriptParser:
    """
    Regex-based JavaScript/TypeScript parser for symbol extraction.
    Extracts functions, classes, interfaces, types, and variables.
    """
    
    # Regex patterns for JS/TS constructs
    PATTERNS = {
        # Function declarations: function name(params) or async function name(params)
        "function": re.compile(
            r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        ),
        # Arrow functions: const name = (params) => or const name = async (params) =>
        "arrow_function": re.compile(
            r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
            re.MULTILINE
        ),
        # Class declarations: class Name or class Name extends Parent
        "class": re.compile(
            r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+[\w,\s]+)?',
            re.MULTILINE
        ),
        # Interface declarations (TypeScript)
        "interface": re.compile(
            r'^(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?',
            re.MULTILINE
        ),
        # Type aliases (TypeScript)
        "type_alias": re.compile(
            r'^(?:export\s+)?type\s+(\w+)\s*=',
            re.MULTILINE
        ),
        # Enum declarations
        "enum": re.compile(
            r'^(?:export\s+)?(?:const\s+)?enum\s+(\w+)',
            re.MULTILINE
        ),
        # ES6 imports
        "import": re.compile(
            r'^import\s+(?:{([^}]+)}|(\w+)|\*\s+as\s+(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]',
            re.MULTILINE
        ),
        # Require imports
        "require": re.compile(
            r'^(?:const|let|var)\s+(?:{([^}]+)}|(\w+))\s*=\s*require\([\'"]([^\'"]+)[\'"]\)',
            re.MULTILINE
        ),
        # Constants (UPPER_CASE)
        "constant": re.compile(
            r'^(?:export\s+)?const\s+([A-Z][A-Z0-9_]+)\s*=',
            re.MULTILINE
        ),
        # Method in class (simplified)
        "method": re.compile(
            r'^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*{',
            re.MULTILINE
        ),
    }
    
    def __init__(self):
        self.current_file = ""
        self.source_lines = []
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a JavaScript/TypeScript file"""
        self.current_file = str(file_path)
        result = ParseResult(file_path=self.current_file)
        
        try:
            content = file_path.read_text(encoding="utf-8")
            self.source_lines = content.split("\n")
            result.total_lines = len(self.source_lines)
            result.code_lines = self._count_code_lines(content)
            
            # Extract imports
            result.imports = self._extract_imports(content)
            
            # Extract symbols
            result.symbols = self._extract_symbols(content)
            
        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")
        
        return result
    
    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines"""
        count = 0
        in_multiline_comment = False
        
        for line in content.split("\n"):
            stripped = line.strip()
            
            # Handle multiline comments
            if "/*" in stripped:
                in_multiline_comment = True
            if "*/" in stripped:
                in_multiline_comment = False
                continue
            
            if in_multiline_comment:
                continue
            
            # Skip empty lines and single-line comments
            if not stripped or stripped.startswith("//"):
                continue
            
            count += 1
        
        return count
    
    def _extract_imports(self, content: str) -> List[ImportInfo]:
        """Extract import statements"""
        imports = []
        
        # ES6 imports
        for match in self.PATTERNS["import"].finditer(content):
            named, default, namespace, module = match.groups()
            names = []
            if named:
                names = [n.strip().split(" as ")[0] for n in named.split(",")]
            elif default:
                names = [default]
            elif namespace:
                names = [f"* as {namespace}"]
            
            imports.append(ImportInfo(
                module=module,
                names=names,
                is_from_import=True,
                line_number=content[:match.start()].count("\n") + 1,
            ))
        
        # Require imports
        for match in self.PATTERNS["require"].finditer(content):
            named, default, module = match.groups()
            names = []
            if named:
                names = [n.strip() for n in named.split(",")]
            elif default:
                names = [default]
            
            imports.append(ImportInfo(
                module=module,
                names=names,
                is_from_import=False,
                line_number=content[:match.start()].count("\n") + 1,
            ))
        
        return imports
    
    def _extract_symbols(self, content: str) -> List[CodeSymbol]:
        """Extract all symbols from JavaScript/TypeScript code"""
        symbols = []
        
        # Functions
        for match in self.PATTERNS["function"].finditer(content):
            name = match.group(1)
            params = match.group(2)
            line_num = content[:match.start()].count("\n") + 1
            
            is_async = "async" in content[max(0, match.start()-20):match.start()]
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.ASYNC_FUNCTION if is_async else SymbolType.FUNCTION,
                file_path=self.current_file,
                line_start=line_num,
                line_end=self._find_block_end(content, match.end()),
                signature=f"function {name}({params})",
                parameters=[p.strip() for p in params.split(",") if p.strip()],
            ))
        
        # Arrow functions
        for match in self.PATTERNS["arrow_function"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.ARROW_FUNCTION,
                file_path=self.current_file,
                line_start=line_num,
                line_end=self._find_block_end(content, match.end()),
                signature=f"const {name} = () =>",
            ))
        
        # Classes
        for match in self.PATTERNS["class"].finditer(content):
            name = match.group(1)
            parent = match.group(2)
            line_num = content[:match.start()].count("\n") + 1
            
            signature = f"class {name}"
            if parent:
                signature += f" extends {parent}"
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.CLASS,
                file_path=self.current_file,
                line_start=line_num,
                line_end=self._find_block_end(content, match.end()),
                signature=signature,
                base_classes=[parent] if parent else [],
            ))
        
        # Interfaces (TypeScript)
        for match in self.PATTERNS["interface"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.INTERFACE,
                file_path=self.current_file,
                line_start=line_num,
                line_end=self._find_block_end(content, match.end()),
                signature=f"interface {name}",
            ))
        
        # Type aliases (TypeScript)
        for match in self.PATTERNS["type_alias"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.TYPE_ALIAS,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"type {name}",
            ))
        
        # Enums
        for match in self.PATTERNS["enum"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.ENUM,
                file_path=self.current_file,
                line_start=line_num,
                line_end=self._find_block_end(content, match.end()),
                signature=f"enum {name}",
            ))
        
        # Constants
        for match in self.PATTERNS["constant"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=name,
                symbol_type=SymbolType.CONSTANT,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"const {name}",
            ))
        
        return symbols
    
    def _find_block_end(self, content: str, start_pos: int) -> int:
        """Find the end line of a code block (balanced braces)"""
        brace_count = 0
        in_string = False
        string_char = None
        line_num = content[:start_pos].count("\n") + 1
        
        i = start_pos
        while i < len(content):
            char = content[i]
            
            # Handle strings
            if char in ('"', "'", '`') and (i == 0 or content[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return content[:i].count("\n") + 1
                elif char == '\n':
                    line_num += 1
            
            i += 1
        
        return line_num


# =============================================================================
# HTML Parser
# =============================================================================

class HTMLParser:
    """
    Regex-based HTML parser for structure extraction.
    Extracts elements, components, scripts, and styles.
    """
    
    PATTERNS = {
        # HTML elements with id
        "id_element": re.compile(r'<(\w+)[^>]*\sid=["\']([^"\']+)["\']', re.IGNORECASE),
        # HTML elements with class
        "class_element": re.compile(r'<(\w+)[^>]*\sclass=["\']([^"\']+)["\']', re.IGNORECASE),
        # Custom components (PascalCase or kebab-case with dash)
        "component": re.compile(r'<([A-Z][\w-]*|[\w]+-[\w-]+)[^>]*>', re.MULTILINE),
        # Script tags
        "script": re.compile(r'<script([^>]*)>(.*?)</script>', re.IGNORECASE | re.DOTALL),
        # Style tags
        "style": re.compile(r'<style([^>]*)>(.*?)</style>', re.IGNORECASE | re.DOTALL),
        # Link tags (for CSS)
        "link": re.compile(r'<link[^>]*href=["\']([^"\']+\.css)["\'][^>]*>', re.IGNORECASE),
        # Meta tags
        "meta": re.compile(r'<meta[^>]*name=["\']([^"\']+)["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE),
        # Forms
        "form": re.compile(r'<form[^>]*(?:id=["\']([^"\']+)["\'])?[^>]*(?:action=["\']([^"\']+)["\'])?', re.IGNORECASE),
    }
    
    def __init__(self):
        self.current_file = ""
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse an HTML file"""
        self.current_file = str(file_path)
        result = ParseResult(file_path=self.current_file)
        
        try:
            content = file_path.read_text(encoding="utf-8")
            result.total_lines = content.count("\n") + 1
            result.code_lines = len([l for l in content.split("\n") if l.strip()])
            
            result.symbols = self._extract_symbols(content)
            result.imports = self._extract_imports(content)
            
        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")
        
        return result
    
    def _extract_imports(self, content: str) -> List[ImportInfo]:
        """Extract linked resources (CSS, JS)"""
        imports = []
        
        # CSS links
        for match in self.PATTERNS["link"].finditer(content):
            href = match.group(1)
            imports.append(ImportInfo(
                module=href,
                names=["stylesheet"],
                is_from_import=True,
                line_number=content[:match.start()].count("\n") + 1,
            ))
        
        # Script sources
        for match in re.finditer(r'<script[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE):
            src = match.group(1)
            imports.append(ImportInfo(
                module=src,
                names=["script"],
                is_from_import=True,
                line_number=content[:match.start()].count("\n") + 1,
            ))
        
        return imports
    
    def _extract_symbols(self, content: str) -> List[CodeSymbol]:
        """Extract symbols from HTML"""
        symbols = []
        
        # Elements with IDs
        for match in self.PATTERNS["id_element"].finditer(content):
            tag, id_value = match.groups()
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=f"#{id_value}",
                symbol_type=SymbolType.HTML_ELEMENT,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"<{tag} id=\"{id_value}\">",
            ))
        
        # Elements with classes (unique classes only)
        seen_classes = set()
        for match in self.PATTERNS["class_element"].finditer(content):
            tag, class_value = match.groups()
            line_num = content[:match.start()].count("\n") + 1
            
            for cls in class_value.split():
                if cls not in seen_classes:
                    seen_classes.add(cls)
                    symbols.append(CodeSymbol(
                        name=f".{cls}",
                        symbol_type=SymbolType.CSS_CLASS,
                        file_path=self.current_file,
                        line_start=line_num,
                        line_end=line_num,
                        signature=f"<{tag} class=\"{cls}\">",
                    ))
        
        # Custom components
        seen_components = set()
        for match in self.PATTERNS["component"].finditer(content):
            component = match.group(1)
            if component not in seen_components and not component.lower() in [
                "div", "span", "p", "a", "img", "ul", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                "table", "tr", "td", "th", "form", "input", "button", "select", "option",
                "header", "footer", "nav", "section", "article", "aside", "main"
            ]:
                seen_components.add(component)
                line_num = content[:match.start()].count("\n") + 1
                
                symbols.append(CodeSymbol(
                    name=component,
                    symbol_type=SymbolType.HTML_COMPONENT,
                    file_path=self.current_file,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"<{component}>",
                ))
        
        # Forms
        for match in self.PATTERNS["form"].finditer(content):
            form_id, action = match.groups()
            line_num = content[:match.start()].count("\n") + 1
            
            name = form_id or action or "form"
            symbols.append(CodeSymbol(
                name=f"form:{name}",
                symbol_type=SymbolType.HTML_ELEMENT,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"<form id=\"{form_id or ''}\" action=\"{action or ''}\">",
            ))
        
        return symbols


# =============================================================================
# CSS Parser
# =============================================================================

class CSSParser:
    """
    Regex-based CSS parser for selector and variable extraction.
    Supports CSS, SCSS, and LESS basics.
    """
    
    PATTERNS = {
        # Class selectors
        "class": re.compile(r'\.([a-zA-Z_][\w-]*)\s*[{,:]', re.MULTILINE),
        # ID selectors
        "id": re.compile(r'#([a-zA-Z_][\w-]*)\s*[{,:]', re.MULTILINE),
        # CSS custom properties (variables)
        "variable": re.compile(r'--([a-zA-Z_][\w-]*)\s*:', re.MULTILINE),
        # @keyframes
        "keyframes": re.compile(r'@keyframes\s+([\w-]+)', re.MULTILINE),
        # @media queries
        "media": re.compile(r'@media\s+([^{]+)', re.MULTILINE),
        # @import
        "import": re.compile(r'@import\s+["\']([^"\']+)["\']', re.MULTILINE),
        # SCSS/LESS mixins
        "mixin": re.compile(r'@mixin\s+([\w-]+)', re.MULTILINE),
        # SCSS/LESS variables
        "scss_variable": re.compile(r'\$([a-zA-Z_][\w-]*)\s*:', re.MULTILINE),
    }
    
    def __init__(self):
        self.current_file = ""
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a CSS/SCSS/LESS file"""
        self.current_file = str(file_path)
        result = ParseResult(file_path=self.current_file)
        
        try:
            content = file_path.read_text(encoding="utf-8")
            result.total_lines = content.count("\n") + 1
            result.code_lines = len([l for l in content.split("\n") if l.strip() and not l.strip().startswith("//")])
            
            result.symbols = self._extract_symbols(content)
            result.imports = self._extract_imports(content)
            
        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")
        
        return result
    
    def _extract_imports(self, content: str) -> List[ImportInfo]:
        """Extract @import statements"""
        imports = []
        
        for match in self.PATTERNS["import"].finditer(content):
            path = match.group(1)
            imports.append(ImportInfo(
                module=path,
                names=["*"],
                is_from_import=True,
                line_number=content[:match.start()].count("\n") + 1,
            ))
        
        return imports
    
    def _extract_symbols(self, content: str) -> List[CodeSymbol]:
        """Extract CSS symbols"""
        symbols = []
        
        # Classes (unique)
        seen_classes = set()
        for match in self.PATTERNS["class"].finditer(content):
            cls = match.group(1)
            if cls not in seen_classes:
                seen_classes.add(cls)
                line_num = content[:match.start()].count("\n") + 1
                
                symbols.append(CodeSymbol(
                    name=f".{cls}",
                    symbol_type=SymbolType.CSS_CLASS,
                    file_path=self.current_file,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f".{cls} {{ }}",
                ))
        
        # IDs (unique)
        seen_ids = set()
        for match in self.PATTERNS["id"].finditer(content):
            id_val = match.group(1)
            if id_val not in seen_ids:
                seen_ids.add(id_val)
                line_num = content[:match.start()].count("\n") + 1
                
                symbols.append(CodeSymbol(
                    name=f"#{id_val}",
                    symbol_type=SymbolType.CSS_ID,
                    file_path=self.current_file,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"#{id_val} {{ }}",
                ))
        
        # CSS Variables
        for match in self.PATTERNS["variable"].finditer(content):
            var_name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=f"--{var_name}",
                symbol_type=SymbolType.CSS_VARIABLE,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"--{var_name}: value;",
            ))
        
        # SCSS/LESS variables
        for match in self.PATTERNS["scss_variable"].finditer(content):
            var_name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=f"${var_name}",
                symbol_type=SymbolType.CSS_VARIABLE,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"${var_name}: value;",
            ))
        
        # Keyframes
        for match in self.PATTERNS["keyframes"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=f"@keyframes {name}",
                symbol_type=SymbolType.CSS_KEYFRAMES,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"@keyframes {name} {{ }}",
            ))
        
        # Media queries (simplified)
        for match in self.PATTERNS["media"].finditer(content):
            query = match.group(1).strip()[:50]  # Truncate long queries
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=f"@media {query}",
                symbol_type=SymbolType.CSS_MEDIA,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"@media {query} {{ }}",
            ))
        
        # Mixins
        for match in self.PATTERNS["mixin"].finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            
            symbols.append(CodeSymbol(
                name=f"@mixin {name}",
                symbol_type=SymbolType.FUNCTION,
                file_path=self.current_file,
                line_start=line_num,
                line_end=line_num,
                signature=f"@mixin {name}() {{ }}",
            ))
        
        return symbols


# =============================================================================
# Multi-Language Parser (Unified Interface)
# =============================================================================

class MultiLanguageParser:
    """
    Unified parser that automatically selects the right parser based on file extension.
    Supports Python, JavaScript, TypeScript, HTML, and CSS.
    """
    
    def __init__(self):
        self.python_parser = PythonASTParser()
        self.js_parser = JavaScriptParser()
        self.html_parser = HTMLParser()
        self.css_parser = CSSParser()
        self.chunker = CodeChunker()
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse any supported file type"""
        language = get_language(file_path)
        
        if language == "python":
            return self.python_parser.parse_file(file_path)
        elif language in ("javascript", "typescript"):
            return self.js_parser.parse_file(file_path)
        elif language == "html":
            return self.html_parser.parse_file(file_path)
        elif language in ("css", "scss", "sass", "less"):
            return self.css_parser.parse_file(file_path)
        else:
            # Return basic result for unsupported files
            return self._parse_unknown(file_path)
    
    def _parse_unknown(self, file_path: Path) -> ParseResult:
        """Basic parsing for unknown file types"""
        result = ParseResult(file_path=str(file_path))
        
        try:
            content = file_path.read_text(encoding="utf-8")
            result.total_lines = content.count("\n") + 1
            result.code_lines = len([l for l in content.split("\n") if l.strip()])
        except Exception as e:
            result.errors.append(f"Could not read file: {e}")
        
        return result
    
    def chunk_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Chunk any supported file type"""
        language = get_language(file_path)
        
        if language == "python":
            return self.chunker.chunk_file(file_path)
        else:
            # For non-Python, use simple chunking
            return self.chunker._simple_chunk(
                file_path.read_text(encoding="utf-8"),
                str(file_path)
            )
    
    def get_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """Get structured overview of any supported file"""
        result = self.parse_file(file_path)
        language = get_language(file_path)
        
        structure = {
            "file": str(file_path),
            "language": language,
            "lines": result.total_lines,
            "code_lines": result.code_lines,
            "imports": len(result.imports),
            "symbols": len(result.symbols),
            "errors": result.errors,
        }
        
        # Add language-specific details
        if language == "python":
            structure["classes"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.CLASS]
            structure["functions"] = [s.name for s in result.symbols if s.symbol_type in (SymbolType.FUNCTION, SymbolType.ASYNC_FUNCTION)]
        elif language in ("javascript", "typescript"):
            structure["classes"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.CLASS]
            structure["functions"] = [s.name for s in result.symbols if s.symbol_type in (SymbolType.FUNCTION, SymbolType.ARROW_FUNCTION)]
            structure["interfaces"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.INTERFACE]
            structure["types"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.TYPE_ALIAS]
        elif language == "html":
            structure["components"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.HTML_COMPONENT]
            structure["ids"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.HTML_ELEMENT]
        elif language in ("css", "scss", "sass", "less"):
            structure["classes"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.CSS_CLASS]
            structure["ids"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.CSS_ID]
            structure["variables"] = [s.name for s in result.symbols if s.symbol_type == SymbolType.CSS_VARIABLE]
        
        return structure


# =============================================================================
# Convenience functions
# =============================================================================

def parse_python_file(file_path: Path) -> ParseResult:
    """Parse a Python file and return results"""
    parser = PythonASTParser()
    return parser.parse_file(file_path)


def parse_file(file_path: Path) -> ParseResult:
    """Parse any supported file type"""
    parser = MultiLanguageParser()
    return parser.parse_file(file_path)


def extract_symbols(file_path: Path) -> List[CodeSymbol]:
    """Extract all symbols from any supported file"""
    result = parse_file(file_path)
    return result.symbols


def get_file_structure(file_path: Path) -> Dict[str, Any]:
    """Get a structured overview of any supported file"""
    parser = MultiLanguageParser()
    return parser.get_file_structure(file_path)



