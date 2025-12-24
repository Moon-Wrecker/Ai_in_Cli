"""
Hybrid Retriever - Combines semantic, keyword, and graph-based search
Uses Reciprocal Rank Fusion (RRF) for optimal result merging
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from storage.chroma_store import ChromaVectorStore, get_vector_store, SearchResult
from storage.graph_store import DependencyGraph, get_dependency_graph, EdgeType
from indexing.ast_indexer import ASTIndexer
from indexing.semantic_indexer import SemanticIndexer


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined scoring"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    combined_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    chunk_type: str = "text"
    symbol_name: Optional[str] = None
    relevance_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "file": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.combined_score, 4),
            "scores": {
                "semantic": round(self.semantic_score, 4),
                "keyword": round(self.keyword_score, 4),
                "graph": round(self.graph_score, 4),
            },
            "type": self.chunk_type,
            "symbol": self.symbol_name,
            "reasons": self.relevance_reasons,
        }


class KeywordSearcher:
    """
    Simple keyword-based search using regex matching.
    Supports BM25-like scoring based on term frequency.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.settings = get_settings()
    
    def search(
        self,
        query: str,
        n_results: int = 20,
        file_filter: Optional[str] = None,
    ) -> List[Tuple[str, int, int, str, float]]:
        """
        Search for keyword matches in files.
        
        Returns:
            List of (file_path, start_line, end_line, content, score)
        """
        results = []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # Build regex patterns
        patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in query_terms]
        
        # Search files
        for file_path in self._get_searchable_files():
            if file_filter and file_filter not in str(file_path):
                continue
            
            file_results = self._search_file(file_path, patterns, query_terms)
            results.extend(file_results)
        
        # Sort by score and limit
        results.sort(key=lambda x: x[4], reverse=True)
        return results[:n_results]
    
    def _get_searchable_files(self) -> List[Path]:
        """Get all searchable files with safety limits"""
        files = []
        allowed_extensions = self.settings.allowed_extensions
        ignore_patterns = self.settings.ignore_patterns
        
        # Additional directories to always skip
        skip_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 
                    '.cache', '.chroma', 'chroma_data', '.mypy_cache', '.pytest_cache'}
        
        MAX_FILES = 500  # Safety limit for search
        
        for file_path in self.sandbox_path.rglob("*"):
            if len(files) >= MAX_FILES:
                break  # Stop to prevent resource exhaustion
                
            if not file_path.is_file():
                continue
            
            # Skip heavy directories
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            
            if file_path.suffix.lower() not in allowed_extensions:
                continue
            
            if any(p in str(file_path) for p in ignore_patterns):
                continue
            
            files.append(file_path)
        
        return files
    
    def _search_file(
        self,
        file_path: Path,
        patterns: List[re.Pattern],
        query_terms: List[str],
    ) -> List[Tuple[str, int, int, str, float]]:
        """Search a single file"""
        try:
            content = file_path.read_text(encoding="utf-8")
        except:
            return []
        
        lines = content.split("\n")
        results = []
        
        # Find matching regions
        context_lines = 3
        matched_lines = set()
        term_matches = defaultdict(set)  # term -> set of line numbers
        
        for i, line in enumerate(lines):
            for j, pattern in enumerate(patterns):
                if pattern.search(line):
                    matched_lines.add(i)
                    term_matches[query_terms[j]].add(i)
        
        if not matched_lines:
            return []
        
        # Group consecutive matches into chunks
        sorted_lines = sorted(matched_lines)
        chunks = []
        current_chunk = [sorted_lines[0]]
        
        for line_num in sorted_lines[1:]:
            if line_num - current_chunk[-1] <= context_lines * 2:
                current_chunk.append(line_num)
            else:
                chunks.append(current_chunk)
                current_chunk = [line_num]
        chunks.append(current_chunk)
        
        # Create results for each chunk
        for chunk in chunks:
            start = max(0, min(chunk) - context_lines)
            end = min(len(lines) - 1, max(chunk) + context_lines)
            
            chunk_content = "\n".join(lines[start:end + 1])
            
            # Calculate BM25-like score
            score = self._calculate_score(
                chunk_content, query_terms, patterns, len(lines)
            )
            
            results.append((
                str(file_path),
                start + 1,  # 1-indexed
                end + 1,
                chunk_content,
                score,
            ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into search terms"""
        # Split on non-alphanumeric, keep underscores for code
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]
    
    def _calculate_score(
        self,
        content: str,
        query_terms: List[str],
        patterns: List[re.Pattern],
        doc_length: int,
    ) -> float:
        """
        Calculate BM25-like relevance score.
        """
        k1 = 1.5  # Term frequency saturation
        b = 0.75  # Length normalization
        avg_doc_length = 100  # Approximate average
        
        score = 0.0
        content_lower = content.lower()
        
        for term, pattern in zip(query_terms, patterns):
            # Term frequency
            tf = len(pattern.findall(content_lower))
            
            if tf > 0:
                # BM25 term frequency component
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                
                # IDF would require corpus statistics, use simplified version
                idf = 1.0  # Simplified
                
                score += tf_component * idf
        
        # Bonus for matching multiple terms
        terms_matched = sum(1 for p in patterns if p.search(content_lower))
        if terms_matched > 1:
            score *= (1 + 0.2 * (terms_matched - 1))
        
        return score


class HybridRetriever:
    """
    Combines semantic search, keyword search, and graph-based retrieval.
    Uses Reciprocal Rank Fusion (RRF) for optimal result merging.
    """
    
    def __init__(
        self,
        sandbox_path: Optional[Path] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        graph: Optional[DependencyGraph] = None,
        persist: bool = True,
    ):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.settings = get_settings()
        self._persist = persist
        
        # Store provided instances or use lazy loading
        self._vector_store = vector_store
        self._graph = graph
        self._keyword_searcher = None
        self._ast_indexer = None
    
    @property
    def vector_store(self) -> ChromaVectorStore:
        """Lazy load vector store"""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store
    
    @property
    def graph(self) -> DependencyGraph:
        """Lazy load graph"""
        if self._graph is None:
            self._graph = get_dependency_graph()
        return self._graph
    
    @property
    def keyword_searcher(self) -> KeywordSearcher:
        """Lazy load keyword searcher"""
        if self._keyword_searcher is None:
            self._keyword_searcher = KeywordSearcher(self.sandbox_path)
        return self._keyword_searcher
    
    @property
    def ast_indexer(self) -> ASTIndexer:
        """Lazy load AST indexer"""
        if self._ast_indexer is None:
            self._ast_indexer = ASTIndexer(self.sandbox_path)
        return self._ast_indexer
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        file_filter: Optional[str] = None,
        search_types: Optional[List[str]] = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining multiple retrieval methods.
        
        Args:
            query: Search query
            n_results: Maximum results to return
            file_filter: Optional file path filter
            search_types: Which search types to use ("semantic", "keyword", "graph")
                         Defaults to all.
        
        Returns:
            List of HybridSearchResult objects
        """
        if search_types is None:
            search_types = ["semantic", "keyword", "graph"]
        
        # Get results from each method
        all_rankings = {}
        
        if "semantic" in search_types:
            semantic_results = self._semantic_search(query, n_results * 2, file_filter)
            all_rankings["semantic"] = semantic_results
        
        if "keyword" in search_types:
            keyword_results = self._keyword_search(query, n_results * 2, file_filter)
            all_rankings["keyword"] = keyword_results
        
        if "graph" in search_types:
            graph_results = self._graph_search(query, n_results * 2)
            all_rankings["graph"] = graph_results
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(all_rankings)
        
        # Convert to HybridSearchResult objects
        results = self._build_results(combined, all_rankings)
        
        return results[:n_results]
    
    def _semantic_search(
        self,
        query: str,
        n_results: int,
        file_filter: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Get semantic search results"""
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_file=file_filter,
        )
        
        # Return (unique_key, score) tuples
        rankings = []
        for r in results:
            key = f"{r.file_path}:{r.start_line}-{r.end_line}"
            rankings.append((key, r.score, r))
        
        return rankings
    
    def _keyword_search(
        self,
        query: str,
        n_results: int,
        file_filter: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Get keyword search results"""
        results = self.keyword_searcher.search(
            query=query,
            n_results=n_results,
            file_filter=file_filter,
        )
        
        rankings = []
        for file_path, start, end, content, score in results:
            key = f"{file_path}:{start}-{end}"
            rankings.append((key, score, (file_path, start, end, content)))
        
        return rankings
    
    def _graph_search(
        self,
        query: str,
        n_results: int,
    ) -> List[Tuple[str, float]]:
        """Get graph-based search results"""
        rankings = []
        
        # Search for symbols matching the query
        from storage.graph_store import NodeType
        
        nodes = self.graph.search_nodes(query, limit=n_results // 2)
        
        for node in nodes:
            # Get related nodes
            related = self.graph.get_related_nodes(
                node.id,
                max_depth=2,
                max_nodes=5,
            )
            
            # Score based on match quality
            score = 1.0 if query.lower() in node.name.lower() else 0.5
            
            key = f"{node.file_path}:{node.line_number}-{node.line_number}"
            rankings.append((key, score, node))
            
            # Add related nodes with lower scores
            for rel in related:
                rel_key = f"{rel['file']}:{rel['line']}-{rel['line']}"
                rel_score = 0.3 / (rel['distance'] + 1)
                rankings.append((rel_key, rel_score, rel))
        
        return rankings
    
    def _reciprocal_rank_fusion(
        self,
        rankings: Dict[str, List[Tuple[str, float, Any]]],
        k: int = 60,
    ) -> Dict[str, float]:
        """
        Combine rankings using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) across all ranking lists
        
        Args:
            rankings: Dict of method -> list of (key, score, data) tuples
            k: Constant to prevent high scores for items ranked first
            
        Returns:
            Dict of key -> combined_score
        """
        weights = {
            "semantic": self.settings.semantic_weight,
            "keyword": self.settings.keyword_weight,
            "graph": self.settings.graph_weight,
        }
        
        combined_scores = defaultdict(float)
        
        for method, results in rankings.items():
            weight = weights.get(method, 0.33)
            
            # Sort by score to get ranks
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            
            for rank, (key, score, data) in enumerate(sorted_results, 1):
                rrf_score = weight / (k + rank)
                combined_scores[key] += rrf_score
        
        return combined_scores
    
    def _build_results(
        self,
        combined_scores: Dict[str, float],
        all_rankings: Dict[str, List[Tuple[str, float, Any]]],
    ) -> List[HybridSearchResult]:
        """Build HybridSearchResult objects from combined scores"""
        
        # Create lookup for original data
        data_lookup = {}
        score_lookup = defaultdict(lambda: {"semantic": 0, "keyword": 0, "graph": 0})
        
        for method, results in all_rankings.items():
            for key, score, data in results:
                if key not in data_lookup:
                    data_lookup[key] = data
                score_lookup[key][method] = score
        
        # Sort by combined score
        sorted_keys = sorted(
            combined_scores.keys(),
            key=lambda k: combined_scores[k],
            reverse=True,
        )
        
        results = []
        for key in sorted_keys:
            data = data_lookup.get(key)
            if data is None:
                continue
            
            scores = score_lookup[key]
            
            # Parse key
            parts = key.split(":")
            file_path = parts[0]
            line_range = parts[1].split("-") if len(parts) > 1 else ["0", "0"]
            
            # Extract content and metadata based on data type
            content = ""
            chunk_type = "text"
            symbol_name = None
            
            if isinstance(data, SearchResult):
                content = data.content
                chunk_type = data.chunk_type
            elif isinstance(data, tuple) and len(data) >= 4:
                content = data[3]
            elif hasattr(data, "signature"):
                content = data.signature
                chunk_type = data.node_type.value if hasattr(data, "node_type") else "symbol"
                symbol_name = data.name if hasattr(data, "name") else None
            elif isinstance(data, dict):
                content = data.get("signature", data.get("name", str(data)))
                symbol_name = data.get("name")
            
            # Build relevance reasons
            reasons = []
            if scores["semantic"] > 0:
                reasons.append("semantic match")
            if scores["keyword"] > 0:
                reasons.append("keyword match")
            if scores["graph"] > 0:
                reasons.append("graph relation")
            
            result = HybridSearchResult(
                content=content,
                file_path=file_path,
                start_line=int(line_range[0]),
                end_line=int(line_range[1]) if len(line_range) > 1 else int(line_range[0]),
                combined_score=combined_scores[key],
                semantic_score=scores["semantic"],
                keyword_score=scores["keyword"],
                graph_score=scores["graph"],
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                relevance_reasons=reasons,
            )
            results.append(result)
        
        return results
    
    def search_symbol(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Search for a specific symbol by name"""
        nodes = self.graph.search_nodes(symbol_name, limit=10)
        
        results = []
        for node in nodes:
            result = node.to_dict()
            
            # Get relationships
            deps = self.graph.get_dependencies(node.id)
            result["dependencies"] = [{"id": d[0], "type": d[1].value} for d in deps[:5]]
            
            callers = self.graph.get_dependents(node.id, edge_types=[EdgeType.CALLS])
            result["callers"] = [{"id": c[0]} for c in callers[:5]]
            
            results.append(result)
        
        return results
    
    def get_context(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 10,
    ) -> Dict[str, Any]:
        """Get code context around a line"""
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            context = {
                "file": file_path,
                "start_line": start + 1,
                "end_line": end,
                "content": "\n".join(lines[start:end]),
                "target_line": line_number,
            }
            
            # Get symbol at this line
            if path.suffix == ".py":
                try:
                    relative = path.relative_to(self.sandbox_path)
                    module_id = ".".join(relative.with_suffix("").parts)
                    
                    # Find symbol at line
                    for node in self.graph.nodes.values():
                        if node.file_path == file_path and node.line_number == line_number:
                            context["symbol"] = node.to_dict()
                            break
                except:
                    pass
            
            return context
            
        except Exception as e:
            return {"error": str(e)}


# Convenience functions
def hybrid_search(
    query: str,
    n_results: int = 10,
    file_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Perform hybrid search"""
    retriever = HybridRetriever()
    results = retriever.search(query, n_results, file_filter)
    return [r.to_dict() for r in results]


def search_symbol(symbol_name: str) -> List[Dict[str, Any]]:
    """Search for a symbol"""
    retriever = HybridRetriever()
    return retriever.search_symbol(symbol_name)



