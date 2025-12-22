"""
Dependency Graph Builder - Analyzes import and call relationships
Creates a comprehensive graph of code dependencies
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from storage.graph_store import (
    DependencyGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    get_dependency_graph,
)
from utils.parsers import PythonASTParser, ParseResult


@dataclass
class DependencyStats:
    """Statistics from dependency analysis"""
    modules_analyzed: int = 0
    import_edges: int = 0
    call_edges: int = 0
    inheritance_edges: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    circular_dependencies: List[List[str]] = None
    
    def __post_init__(self):
        if self.circular_dependencies is None:
            self.circular_dependencies = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "modules_analyzed": self.modules_analyzed,
            "import_edges": self.import_edges,
            "call_edges": self.call_edges,
            "inheritance_edges": self.inheritance_edges,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "circular_dependencies": self.circular_dependencies,
        }


class CallGraphVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts function/method call relationships.
    Tracks what functions call what other functions.
    """
    
    def __init__(self, module_id: str, parse_result: ParseResult):
        self.module_id = module_id
        self.parse_result = parse_result
        self.calls: List[Tuple[str, str, int]] = []  # (caller, callee, line)
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        
        # Build symbol lookup
        self.local_symbols = {s.name for s in parse_result.symbols}
        self.import_map = self._build_import_map()
    
    def _build_import_map(self) -> Dict[str, str]:
        """Map imported names to their full module paths"""
        mapping = {}
        for imp in self.parse_result.imports:
            if imp.is_from_import:
                for name in imp.names:
                    if name != "*":
                        mapping[name] = f"{imp.module}.{name}"
            else:
                name = imp.alias or imp.module.split(".")[-1]
                mapping[name] = imp.module
        return mapping
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track current class context"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current function and visit its body"""
        self._visit_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Track async function"""
        self._visit_function(node)
    
    def _visit_function(self, node):
        """Common logic for function visiting"""
        old_function = self.current_function
        
        if self.current_class:
            self.current_function = f"{self.current_class}.{node.name}"
        else:
            self.current_function = node.name
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node: ast.Call):
        """Record function calls"""
        if self.current_function is None:
            # Module-level call
            caller = self.module_id
        else:
            caller = f"{self.module_id}.{self.current_function}"
        
        # Get callee name
        callee = self._get_call_name(node.func)
        
        if callee:
            self.calls.append((caller, callee, node.lineno))
        
        self.generic_visit(node)
    
    def _get_call_name(self, node: ast.AST) -> Optional[str]:
        """Extract the name of a called function"""
        if isinstance(node, ast.Name):
            name = node.id
            # Check if it's imported
            if name in self.import_map:
                return self.import_map[name]
            # Check if it's local
            if name in self.local_symbols:
                return f"{self.module_id}.{name}"
            # Might be builtin or undefined
            return name
        
        elif isinstance(node, ast.Attribute):
            # Something like obj.method()
            value_name = self._get_value_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        
        elif isinstance(node, ast.Subscript):
            # Something like func_dict['key']()
            return self._get_call_name(node.value)
        
        return None
    
    def _get_value_name(self, node: ast.AST) -> Optional[str]:
        """Get the name of a value (for attribute access)"""
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.import_map:
                return self.import_map[name]
            if name == "self" and self.current_class:
                return f"{self.module_id}.{self.current_class}"
            return name
        elif isinstance(node, ast.Attribute):
            value_name = self._get_value_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
        return None


class DependencyGraphBuilder:
    """
    Builds comprehensive dependency graphs for Python codebases.
    Analyzes imports, function calls, and class inheritance.
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
    
    def build_graph(self, force: bool = False) -> DependencyStats:
        """
        Build dependency graph for all Python files in sandbox.
        
        Args:
            force: If True, rebuild the entire graph
            
        Returns:
            DependencyStats with results
        """
        stats = DependencyStats()
        
        if force:
            self.graph.clear()
        
        # Find all Python files
        python_files = list(self.sandbox_path.rglob("*.py"))
        
        # Filter out ignored patterns
        ignore_patterns = self.settings.ignore_patterns
        python_files = [
            f for f in python_files
            if not any(p in str(f) for p in ignore_patterns)
        ]
        
        # First pass: add all modules and symbols
        for file_path in python_files:
            try:
                self._add_file_to_graph(file_path)
                stats.modules_analyzed += 1
            except Exception as e:
                pass  # Skip files with errors
        
        # Second pass: analyze calls (requires all symbols to be present)
        for file_path in python_files:
            try:
                call_count = self._analyze_calls(file_path)
                stats.call_edges += call_count
            except Exception as e:
                pass
        
        # Count edges by type
        graph_stats = self.graph.get_stats()
        edge_types = graph_stats.get("edge_types", {})
        
        stats.import_edges = edge_types.get(EdgeType.IMPORTS.value, 0)
        stats.inheritance_edges = edge_types.get(EdgeType.INHERITS.value, 0)
        stats.total_nodes = graph_stats.get("total_nodes", 0)
        stats.total_edges = graph_stats.get("total_edges", 0)
        
        # Detect circular dependencies
        stats.circular_dependencies = self._find_circular_dependencies()
        
        # Save the graph
        self.graph.save()
        
        return stats
    
    def _add_file_to_graph(self, file_path: Path):
        """Add a file's symbols and imports to the graph"""
        parse_result = self.parser.parse_file(file_path)
        
        if parse_result.errors:
            return
        
        module_id = self._get_module_id(file_path)
        
        # Add module node
        module_node = GraphNode(
            id=module_id,
            name=file_path.stem,
            node_type=NodeType.MODULE,
            file_path=str(file_path),
            line_number=1,
            docstring=parse_result.module_docstring,
        )
        self.graph.add_node(module_node)
        
        # Add symbols
        for symbol in parse_result.symbols:
            self._add_symbol(symbol, module_id)
        
        # Add import edges
        for imp in parse_result.imports:
            self._add_import_edge(imp, module_id)
        
        # Mark file as indexed
        self.graph.mark_file_indexed(str(file_path))
    
    def _add_symbol(self, symbol, module_id: str):
        """Add a symbol to the graph"""
        from utils.parsers import SymbolType
        
        if symbol.parent:
            symbol_id = f"{module_id}.{symbol.parent}.{symbol.name}"
            parent_id = f"{module_id}.{symbol.parent}"
        else:
            symbol_id = f"{module_id}.{symbol.name}"
            parent_id = module_id
        
        # Map symbol type
        type_map = {
            SymbolType.CLASS: NodeType.CLASS,
            SymbolType.FUNCTION: NodeType.FUNCTION,
            SymbolType.ASYNC_FUNCTION: NodeType.FUNCTION,
            SymbolType.METHOD: NodeType.METHOD,
            SymbolType.VARIABLE: NodeType.VARIABLE,
            SymbolType.CONSTANT: NodeType.VARIABLE,
        }
        node_type = type_map.get(symbol.symbol_type, NodeType.VARIABLE)
        
        node = GraphNode(
            id=symbol_id,
            name=symbol.name,
            node_type=node_type,
            file_path=symbol.file_path,
            line_number=symbol.line_start,
            signature=symbol.signature,
            docstring=symbol.docstring,
        )
        self.graph.add_node(node)
        
        # Add containment edge
        edge = GraphEdge(
            source=parent_id,
            target=symbol_id,
            edge_type=EdgeType.CONTAINS,
        )
        self.graph.add_edge(edge)
        
        # Add inheritance edges
        if symbol.base_classes:
            for base in symbol.base_classes:
                edge = GraphEdge(
                    source=symbol_id,
                    target=base,  # Might be external
                    edge_type=EdgeType.INHERITS,
                )
                self.graph.add_edge(edge)
    
    def _add_import_edge(self, import_info, module_id: str):
        """Add import edge to graph"""
        if import_info.is_from_import:
            for name in import_info.names:
                if name == "*":
                    target = import_info.module
                else:
                    target = f"{import_info.module}.{name}"
                
                edge = GraphEdge(
                    source=module_id,
                    target=target,
                    edge_type=EdgeType.IMPORTS,
                    metadata={"line": import_info.line_number},
                )
                self.graph.add_edge(edge)
        else:
            edge = GraphEdge(
                source=module_id,
                target=import_info.module,
                edge_type=EdgeType.IMPORTS,
                metadata={"line": import_info.line_number},
            )
            self.graph.add_edge(edge)
    
    def _analyze_calls(self, file_path: Path) -> int:
        """Analyze function calls in a file"""
        parse_result = self.parser.parse_file(file_path)
        
        if parse_result.errors:
            return 0
        
        module_id = self._get_module_id(file_path)
        
        # Parse AST and extract calls
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except:
            return 0
        
        visitor = CallGraphVisitor(module_id, parse_result)
        visitor.visit(tree)
        
        # Add call edges
        call_count = 0
        for caller, callee, line in visitor.calls:
            edge = GraphEdge(
                source=caller,
                target=callee,
                edge_type=EdgeType.CALLS,
                metadata={"line": line},
            )
            if self.graph.add_edge(edge):
                call_count += 1
        
        return call_count
    
    def _get_module_id(self, file_path: Path) -> str:
        """Get module ID from file path"""
        try:
            relative = file_path.relative_to(self.sandbox_path)
            parts = list(relative.parts)
            if parts[-1].endswith(".py"):
                parts[-1] = parts[-1][:-3]
            return ".".join(parts)
        except ValueError:
            return file_path.stem
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in import graph"""
        import networkx as nx
        
        # Create subgraph with only import edges
        import_edges = [
            (u, v) for u, v, d in self.graph.graph.edges(data=True)
            if d.get("type") == EdgeType.IMPORTS.value
        ]
        
        import_graph = nx.DiGraph(import_edges)
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(import_graph))
            # Return first few cycles
            return cycles[:10]
        except:
            return []
    
    def get_import_tree(self, module_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get import tree for a module"""
        return self._build_tree(module_id, EdgeType.IMPORTS, max_depth)
    
    def get_caller_tree(self, symbol_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get callers tree for a symbol"""
        return self._build_reverse_tree(symbol_id, EdgeType.CALLS, max_depth)
    
    def _build_tree(
        self,
        node_id: str,
        edge_type: EdgeType,
        max_depth: int,
        visited: Set[str] = None,
    ) -> Dict[str, Any]:
        """Build dependency tree"""
        if visited is None:
            visited = set()
        
        if node_id in visited or max_depth <= 0:
            return {"id": node_id, "circular": node_id in visited, "children": []}
        
        visited.add(node_id)
        
        node = self.graph.get_node(node_id)
        result = {
            "id": node_id,
            "name": node.name if node else node_id.split(".")[-1],
            "type": node.node_type.value if node else "unknown",
            "children": [],
        }
        
        # Get dependencies
        deps = self.graph.get_dependencies(node_id, edge_types=[edge_type])
        
        for dep_id, _ in deps:
            child = self._build_tree(dep_id, edge_type, max_depth - 1, visited.copy())
            result["children"].append(child)
        
        return result
    
    def _build_reverse_tree(
        self,
        node_id: str,
        edge_type: EdgeType,
        max_depth: int,
        visited: Set[str] = None,
    ) -> Dict[str, Any]:
        """Build reverse dependency tree (who uses this)"""
        if visited is None:
            visited = set()
        
        if node_id in visited or max_depth <= 0:
            return {"id": node_id, "circular": node_id in visited, "callers": []}
        
        visited.add(node_id)
        
        node = self.graph.get_node(node_id)
        result = {
            "id": node_id,
            "name": node.name if node else node_id.split(".")[-1],
            "type": node.node_type.value if node else "unknown",
            "callers": [],
        }
        
        # Get dependents
        deps = self.graph.get_dependents(node_id, edge_types=[edge_type])
        
        for dep_id, _ in deps:
            caller = self._build_reverse_tree(dep_id, edge_type, max_depth - 1, visited.copy())
            result["callers"].append(caller)
        
        return result
    
    def find_symbol(self, name: str) -> List[Dict[str, Any]]:
        """Find symbols by name"""
        nodes = self.graph.search_nodes(name, limit=20)
        return [n.to_dict() for n in nodes]


# Convenience function
def build_dependency_graph(force: bool = False) -> DependencyStats:
    """Build dependency graph for sandbox"""
    builder = DependencyGraphBuilder()
    return builder.build_graph(force=force)



