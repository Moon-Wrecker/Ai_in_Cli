"""
Search Tools - Hybrid search capabilities
Combines semantic, keyword, and graph-based search
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from indexing.hybrid_retriever import HybridRetriever, hybrid_search as _hybrid_search
from indexing.ast_indexer import ASTIndexer
from indexing.semantic_indexer import SemanticIndexer
from storage.graph_store import get_dependency_graph


class SearchTools:
    """
    Provides advanced search capabilities using hybrid retrieval.
    Combines semantic search, keyword matching, and graph analysis.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.settings = get_settings()
        # Lazy initialization - don't load heavy components until needed
        self._retriever = None
        self._graph = None
    
    @property
    def retriever(self) -> HybridRetriever:
        """Lazy load the hybrid retriever"""
        if self._retriever is None:
            self._retriever = HybridRetriever(sandbox_path=self.sandbox_path)
        return self._retriever
    
    @property
    def graph(self):
        """Lazy load the dependency graph"""
        if self._graph is None:
            self._graph = get_dependency_graph()
        return self._graph
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        file_filter: Optional[str] = None,
        search_type: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Search the codebase using hybrid retrieval.
        
        Args:
            query: Search query
            n_results: Maximum results
            file_filter: Filter by file path substring
            search_type: "hybrid", "semantic", "keyword", or "graph"
            
        Returns:
            Dict with search results
        """
        # Map search type to retrieval methods
        if search_type == "hybrid":
            search_types = ["semantic", "keyword", "graph"]
        elif search_type in ["semantic", "keyword", "graph"]:
            search_types = [search_type]
        else:
            search_types = ["semantic", "keyword", "graph"]
        
        try:
            results = self.retriever.search(
                query=query,
                n_results=n_results,
                file_filter=file_filter,
                search_types=search_types,
            )
            
            return {
                "query": query,
                "search_type": search_type,
                "result_count": len(results),
                "results": [r.to_dict() for r in results],
            }
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def search_symbol(self, symbol_name: str) -> Dict[str, Any]:
        """
        Search for a code symbol by name.
        
        Args:
            symbol_name: Symbol name to search for
            
        Returns:
            Dict with matching symbols and their relationships
        """
        try:
            results = self.retriever.search_symbol(symbol_name)
            
            return {
                "query": symbol_name,
                "result_count": len(results),
                "symbols": results,
            }
            
        except Exception as e:
            return {"error": f"Symbol search failed: {str(e)}"}
    
    def get_file_context(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 10,
    ) -> Dict[str, Any]:
        """
        Get code context around a specific line.
        
        Args:
            file_path: Path to file
            line_number: Target line number
            context_lines: Lines of context before/after
            
        Returns:
            Dict with code context and related information
        """
        try:
            return self.retriever.get_context(
                file_path=file_path,
                line_number=line_number,
                context_lines=context_lines,
            )
        except Exception as e:
            return {"error": f"Failed to get context: {str(e)}"}
    
    def get_related_code(
        self,
        file_path: str,
        symbol_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get code related to a file or symbol.
        
        Args:
            file_path: Path to file
            symbol_name: Optional symbol to focus on
            
        Returns:
            Dict with related code
        """
        try:
            # Get the module graph for the file
            resolved = self.sandbox_path / file_path
            if not resolved.exists():
                return {"error": f"File not found: {file_path}"}
            
            module_graph = self.graph.get_module_graph(str(resolved))
            
            result = {
                "file": file_path,
                "module_graph": module_graph,
                "related_files": [],
            }
            
            # Find files that import this one
            try:
                relative = resolved.relative_to(self.sandbox_path)
                module_id = ".".join(relative.with_suffix("").parts)
                
                dependents = self.graph.get_dependents(module_id)
                result["imported_by"] = [d[0] for d in dependents[:10]]
                
                dependencies = self.graph.get_dependencies(module_id)
                result["imports"] = [d[0] for d in dependencies[:10]]
            except:
                pass
            
            # If symbol specified, get its relationships
            if symbol_name:
                symbol_results = self.search_symbol(symbol_name)
                result["symbol_info"] = symbol_results
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to get related code: {str(e)}"}
    
    def search_definitions(
        self,
        name: str,
        definition_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for definitions (functions, classes, etc.).
        
        Args:
            name: Name to search for
            definition_type: "function", "class", "method", or None for all
            
        Returns:
            Dict with matching definitions
        """
        from storage.graph_store import NodeType
        
        # Map type filter
        node_types = None
        if definition_type:
            type_map = {
                "function": [NodeType.FUNCTION],
                "class": [NodeType.CLASS],
                "method": [NodeType.METHOD],
                "variable": [NodeType.VARIABLE],
            }
            node_types = type_map.get(definition_type.lower())
        
        nodes = self.graph.search_nodes(name, node_types=node_types, limit=20)
        
        return {
            "query": name,
            "type_filter": definition_type,
            "result_count": len(nodes),
            "definitions": [n.to_dict() for n in nodes],
        }
    
    def get_call_graph(self, function_name: str) -> Dict[str, Any]:
        """
        Get call graph for a function.
        
        Args:
            function_name: Function name
            
        Returns:
            Dict with callers and callees
        """
        from indexing.dependency_graph import DependencyGraphBuilder
        
        builder = DependencyGraphBuilder(sandbox_path=self.sandbox_path)
        
        # Find the symbol
        nodes = self.graph.search_nodes(function_name, limit=5)
        
        if not nodes:
            return {
                "error": f"Function '{function_name}' not found",
                "suggestion": "Try indexing the workspace first",
            }
        
        results = []
        for node in nodes:
            call_tree = builder.get_caller_tree(node.id, max_depth=2)
            results.append({
                "symbol": node.to_dict(),
                "call_tree": call_tree,
            })
        
        return {
            "query": function_name,
            "results": results,
        }
    
    def get_workspace_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the indexed workspace.
        
        Returns:
            Dict with workspace statistics
        """
        try:
            # Get graph stats
            graph_stats = self.graph.get_stats()
            
            # Get file statistics
            from storage.chroma_store import get_vector_store
            vector_store = get_vector_store()
            vector_stats = vector_store.get_stats()
            
            # Count files by extension with limits to prevent hanging
            extension_counts = {}
            python_files = 0
            total_files = 0
            total_dirs = 0
            
            # Directories to skip
            skip_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 
                        '.cache', '.chroma', 'chroma_data', '.mypy_cache', '.pytest_cache'}
            
            MAX_FILES_TO_COUNT = 1000  # Safety limit
            
            for file_path in self.sandbox_path.rglob("*"):
                # Skip heavy directories
                if any(skip in file_path.parts for skip in skip_dirs):
                    continue
                    
                if file_path.is_dir():
                    total_dirs += 1
                    continue
                    
                if file_path.is_file():
                    total_files += 1
                    if total_files > MAX_FILES_TO_COUNT:
                        break  # Stop counting to prevent hanging
                        
                    ext = file_path.suffix or "(no ext)"
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                    if ext == ".py":
                        python_files += 1
            
            return {
                "sandbox_path": str(self.sandbox_path),
                "graph_stats": graph_stats,
                "vector_stats": vector_stats,
                "total_files": total_files,
                "total_directories": total_dirs,
                "file_stats": {
                    "total_files": total_files,
                    "python_files": python_files,
                    "truncated": total_files > MAX_FILES_TO_COUNT,
                    "by_extension": dict(sorted(
                        extension_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]),
                },
            }
            
        except Exception as e:
            return {"error": f"Failed to get overview: {str(e)}"}
    
    def index_workspace(self, force: bool = False) -> Dict[str, Any]:
        """
        Index the workspace for search.
        
        Args:
            force: If True, re-index everything
            
        Returns:
            Dict with indexing statistics
        """
        try:
            # Run AST indexing
            ast_indexer = ASTIndexer(sandbox_path=self.sandbox_path)
            ast_stats = ast_indexer.index_workspace(force=force)
            
            # Run semantic indexing
            semantic_indexer = SemanticIndexer(sandbox_path=self.sandbox_path)
            semantic_stats = semantic_indexer.index_workspace(force=force)
            
            # Build dependency graph
            from indexing.dependency_graph import DependencyGraphBuilder
            graph_builder = DependencyGraphBuilder(sandbox_path=self.sandbox_path)
            graph_stats = graph_builder.build_graph(force=force)
            
            return {
                "success": True,
                "ast_indexing": ast_stats.to_dict(),
                "semantic_indexing": semantic_stats.to_dict(),
                "dependency_graph": graph_stats.to_dict(),
            }
            
        except Exception as e:
            return {"error": f"Indexing failed: {str(e)}"}


# Tool function wrappers for agent
def search_codebase(query: str, n_results: int = 10) -> str:
    """Search the codebase using hybrid retrieval"""
    tools = SearchTools()
    result = tools.search(query, n_results)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    if not result["results"]:
        return f"No results found for: {query}"
    
    output = [f"Found {result['result_count']} results for '{query}':\n"]
    
    for i, r in enumerate(result["results"][:10], 1):
        output.append(f"{i}. {r['file']}:{r['start_line']}-{r['end_line']}")
        output.append(f"   Score: {r['score']:.3f} ({', '.join(r['reasons'])})")
        # Show preview
        preview = r["content"][:200].replace("\n", " ")
        if len(r["content"]) > 200:
            preview += "..."
        output.append(f"   {preview}\n")
    
    return "\n".join(output)


def search_symbol(symbol_name: str) -> str:
    """Search for a code symbol"""
    tools = SearchTools()
    result = tools.search_symbol(symbol_name)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    if not result["symbols"]:
        return f"No symbols found matching: {symbol_name}"
    
    output = [f"Found {result['result_count']} symbols matching '{symbol_name}':\n"]
    
    for sym in result["symbols"]:
        output.append(f"• {sym['name']} ({sym['type']})")
        output.append(f"  File: {sym['file']}:{sym['line']}")
        if sym.get("signature"):
            output.append(f"  Signature: {sym['signature']}")
        output.append("")
    
    return "\n".join(output)


def get_workspace_overview() -> str:
    """Get workspace overview and statistics"""
    tools = SearchTools()
    result = tools.get_workspace_overview()
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return json.dumps(result, indent=2)


def index_workspace(force: bool = False) -> str:
    """Index the workspace for search"""
    tools = SearchTools()
    result = tools.index_workspace(force)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    output = ["Workspace indexed successfully!\n"]
    
    ast = result.get("ast_indexing", {})
    output.append(f"AST Indexing:")
    output.append(f"  Files processed: {ast.get('files_processed', 0)}")
    output.append(f"  Symbols found: {ast.get('symbols_found', 0)}")
    
    semantic = result.get("semantic_indexing", {})
    output.append(f"\nSemantic Indexing:")
    output.append(f"  Chunks indexed: {semantic.get('chunks_indexed', 0)}")
    
    graph = result.get("dependency_graph", {})
    output.append(f"\nDependency Graph:")
    output.append(f"  Nodes: {graph.get('total_nodes', 0)}")
    output.append(f"  Edges: {graph.get('total_edges', 0)}")
    
    return "\n".join(output)



