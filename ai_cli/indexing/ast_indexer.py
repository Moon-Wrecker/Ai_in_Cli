"""
AST Indexer - Symbol-aware code indexing using Python AST
Provides deep code understanding through structural analysis
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from utils.parsers import (
    PythonASTParser,
    CodeChunker,
    ParseResult,
    CodeSymbol,
    SymbolType,
)
from storage.graph_store import (
    DependencyGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    get_dependency_graph,
)


@dataclass
class IndexStats:
    """Statistics from indexing operation"""
    files_processed: int = 0
    files_skipped: int = 0
    symbols_found: int = 0
    chunks_created: int = 0
    errors: List[str] = None
    duration_seconds: float = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "symbols_found": self.symbols_found,
            "chunks_created": self.chunks_created,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class ASTIndexer:
    """
    Indexes Python code using AST analysis.
    Extracts symbols, builds dependency graphs, and creates semantic chunks.
    """
    
    def __init__(
        self,
        sandbox_path: Optional[Path] = None,
        graph: Optional[DependencyGraph] = None,
    ):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.graph = graph or get_dependency_graph()
        self.settings = get_settings()
        
        self.parser = PythonASTParser()
        self.chunker = CodeChunker(
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        
        # Track what's been indexed
        self.indexed_files: Set[str] = set()
    
    def index_workspace(self, force: bool = False) -> IndexStats:
        """
        Index all Python files in the sandbox workspace.
        
        Args:
            force: If True, re-index all files even if already indexed
            
        Returns:
            IndexStats with results
        """
        start_time = time.time()
        stats = IndexStats()
        
        # Find all Python files
        python_files = self._find_python_files()
        
        for file_path in python_files:
            try:
                # Check if already indexed
                if not force and self.graph.is_file_indexed(str(file_path)):
                    stats.files_skipped += 1
                    continue
                
                # Index the file
                file_stats = self.index_file(file_path)
                stats.files_processed += 1
                stats.symbols_found += file_stats.get("symbols", 0)
                stats.chunks_created += file_stats.get("chunks", 0)
                
                if file_stats.get("errors"):
                    stats.errors.extend(file_stats["errors"])
                
            except Exception as e:
                stats.errors.append(f"Error indexing {file_path}: {str(e)}")
        
        # Save graph after indexing
        self.graph.save()
        
        stats.duration_seconds = time.time() - start_time
        return stats
    
    def index_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Index a single Python file.
        
        Returns:
            Dictionary with indexing results
        """
        result = {
            "file": str(file_path),
            "symbols": 0,
            "chunks": 0,
            "errors": [],
        }
        
        # Remove existing data for this file (re-index)
        self.graph.remove_file(str(file_path))
        
        # Parse the file
        parse_result = self.parser.parse_file(file_path)
        
        if parse_result.errors:
            result["errors"] = parse_result.errors
            return result
        
        # Create module node
        module_id = self._get_module_id(file_path)
        module_node = GraphNode(
            id=module_id,
            name=file_path.stem,
            node_type=NodeType.MODULE,
            file_path=str(file_path),
            line_number=1,
            docstring=parse_result.module_docstring,
        )
        self.graph.add_node(module_node)
        
        # Process symbols
        for symbol in parse_result.symbols:
            self._add_symbol_to_graph(symbol, module_id, parse_result)
            result["symbols"] += 1
        
        # Process imports to build dependency edges
        self._process_imports(parse_result, module_id)
        
        # Mark file as indexed
        self.graph.mark_file_indexed(str(file_path))
        self.indexed_files.add(str(file_path))
        
        # Create chunks for semantic search
        chunks = self.chunker.chunk_file(file_path)
        result["chunks"] = len(chunks)
        
        return result
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in sandbox"""
        python_files = []
        ignore_patterns = self.settings.ignore_patterns
        
        for root, dirs, files in os.walk(self.sandbox_path):
            # Filter directories
            dirs[:] = [
                d for d in dirs
                if not any(
                    pattern in d or d.startswith(".")
                    for pattern in ignore_patterns
                )
            ]
            
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    
                    # Check file size
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        if size_mb <= self.settings.max_file_size_mb:
                            python_files.append(file_path)
                    except:
                        pass
        
        return python_files
    
    def _get_module_id(self, file_path: Path) -> str:
        """Generate module ID from file path"""
        try:
            relative = file_path.relative_to(self.sandbox_path)
            # Convert path to module notation
            parts = list(relative.parts)
            if parts[-1].endswith(".py"):
                parts[-1] = parts[-1][:-3]
            return ".".join(parts)
        except ValueError:
            return file_path.stem
    
    def _add_symbol_to_graph(
        self,
        symbol: CodeSymbol,
        module_id: str,
        parse_result: ParseResult,
    ):
        """Add a symbol and its relationships to the graph"""
        
        # Build symbol ID
        if symbol.parent:
            symbol_id = f"{module_id}.{symbol.parent}.{symbol.name}"
        else:
            symbol_id = f"{module_id}.{symbol.name}"
        
        # Map symbol type to node type
        node_type_map = {
            SymbolType.CLASS: NodeType.CLASS,
            SymbolType.FUNCTION: NodeType.FUNCTION,
            SymbolType.ASYNC_FUNCTION: NodeType.FUNCTION,
            SymbolType.METHOD: NodeType.METHOD,
            SymbolType.VARIABLE: NodeType.VARIABLE,
            SymbolType.CONSTANT: NodeType.VARIABLE,
        }
        node_type = node_type_map.get(symbol.symbol_type, NodeType.VARIABLE)
        
        # Create node
        node = GraphNode(
            id=symbol_id,
            name=symbol.name,
            node_type=node_type,
            file_path=symbol.file_path,
            line_number=symbol.line_start,
            signature=symbol.signature,
            docstring=symbol.docstring,
            metadata={
                "decorators": symbol.decorators,
                "parameters": symbol.parameters,
                "return_type": symbol.return_type,
                "base_classes": symbol.base_classes,
                "complexity": symbol.complexity,
            },
        )
        self.graph.add_node(node)
        
        # Add containment edge (module/class contains this symbol)
        if symbol.parent:
            parent_id = f"{module_id}.{symbol.parent}"
            edge = GraphEdge(
                source=parent_id,
                target=symbol_id,
                edge_type=EdgeType.CONTAINS,
            )
            self.graph.add_edge(edge)
        else:
            edge = GraphEdge(
                source=module_id,
                target=symbol_id,
                edge_type=EdgeType.CONTAINS,
            )
            self.graph.add_edge(edge)
        
        # Add inheritance edges for classes
        if symbol.symbol_type == SymbolType.CLASS and symbol.base_classes:
            for base in symbol.base_classes:
                # Try to resolve base class
                base_id = self._resolve_symbol(base, parse_result)
                edge = GraphEdge(
                    source=symbol_id,
                    target=base_id,
                    edge_type=EdgeType.INHERITS,
                )
                self.graph.add_edge(edge)
        
        # Add decorator edges
        for decorator in symbol.decorators:
            decorator_id = self._resolve_symbol(decorator, parse_result)
            edge = GraphEdge(
                source=decorator_id,
                target=symbol_id,
                edge_type=EdgeType.DECORATES,
            )
            self.graph.add_edge(edge)
    
    def _process_imports(self, parse_result: ParseResult, module_id: str):
        """Process imports and create import edges"""
        for import_info in parse_result.imports:
            if import_info.is_from_import:
                # from X import Y
                for name in import_info.names:
                    if name == "*":
                        target_id = import_info.module
                    else:
                        target_id = f"{import_info.module}.{name}"
                    
                    edge = GraphEdge(
                        source=module_id,
                        target=target_id,
                        edge_type=EdgeType.IMPORTS,
                        metadata={"line": import_info.line_number},
                    )
                    self.graph.add_edge(edge)
            else:
                # import X
                edge = GraphEdge(
                    source=module_id,
                    target=import_info.module,
                    edge_type=EdgeType.IMPORTS,
                    metadata={"line": import_info.line_number},
                )
                self.graph.add_edge(edge)
    
    def _resolve_symbol(self, name: str, parse_result: ParseResult) -> str:
        """Try to resolve a symbol name to its full ID"""
        # Check if it's an imported name
        for import_info in parse_result.imports:
            if name in import_info.names:
                if import_info.is_from_import:
                    return f"{import_info.module}.{name}"
                return name
            
            # Check aliases
            if import_info.alias == name:
                return import_info.module
        
        # Check if it's a local symbol
        for symbol in parse_result.symbols:
            if symbol.name == name:
                return name
        
        # Return as-is (might be builtin or external)
        return name
    
    def get_file_symbols(self, file_path: Path) -> List[CodeSymbol]:
        """Get all symbols from a file"""
        result = self.parser.parse_file(file_path)
        return result.symbols
    
    def get_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """Get structured overview of a file"""
        result = self.parser.parse_file(file_path)
        
        return {
            "file": str(file_path),
            "module_docstring": result.module_docstring,
            "total_lines": result.total_lines,
            "code_lines": result.code_lines,
            "imports": [i.to_dict() for i in result.imports],
            "exports": result.exports,
            "classes": [
                s.to_dict() for s in result.symbols
                if s.symbol_type == SymbolType.CLASS
            ],
            "functions": [
                s.to_dict() for s in result.symbols
                if s.symbol_type in (SymbolType.FUNCTION, SymbolType.ASYNC_FUNCTION)
            ],
            "variables": [
                s.to_dict() for s in result.symbols
                if s.symbol_type in (SymbolType.VARIABLE, SymbolType.CONSTANT)
            ],
            "errors": result.errors,
        }
    
    def get_chunks(self, file_path: Path) -> List[Dict[str, Any]]:
        """Get semantic chunks for a file"""
        return self.chunker.chunk_file(file_path)


# Convenience function
def index_sandbox(force: bool = False) -> IndexStats:
    """Index the entire sandbox workspace"""
    indexer = ASTIndexer()
    return indexer.index_workspace(force=force)



