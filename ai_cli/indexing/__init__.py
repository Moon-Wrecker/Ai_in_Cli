"""
Indexing modules for AI CLI
"""

from .ast_indexer import (
    IndexStats,
    ASTIndexer,
    index_sandbox,
)

from .semantic_indexer import (
    SemanticIndexStats,
    SemanticIndexer,
    index_semantic,
)

from .dependency_graph import (
    DependencyStats,
    CallGraphVisitor,
    DependencyGraphBuilder,
    build_dependency_graph,
)

from .hybrid_retriever import (
    HybridSearchResult,
    KeywordSearcher,
    HybridRetriever,
    hybrid_search,
    search_symbol,
)

from .index_manager import (
    FileState,
    IndexState,
    IndexManager,
    get_index_manager,
    on_file_created,
    on_file_modified,
    on_file_deleted,
)

__all__ = [
    # AST Indexer
    "IndexStats",
    "ASTIndexer",
    "index_sandbox",
    # Semantic Indexer
    "SemanticIndexStats",
    "SemanticIndexer",
    "index_semantic",
    # Dependency Graph
    "DependencyStats",
    "CallGraphVisitor",
    "DependencyGraphBuilder",
    "build_dependency_graph",
    # Hybrid Retriever
    "HybridSearchResult",
    "KeywordSearcher",
    "HybridRetriever",
    "hybrid_search",
    "search_symbol",
    # Index Manager
    "FileState",
    "IndexState",
    "IndexManager",
    "get_index_manager",
    "on_file_created",
    "on_file_modified",
    "on_file_deleted",
]



