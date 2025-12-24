"""
Graph Store for dependency and call graphs
Uses NetworkX for graph operations with JSON persistence
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings, get_cache_path


class EdgeType(Enum):
    """Types of edges in the dependency graph"""
    IMPORTS = "imports"           # Module imports another module
    CALLS = "calls"               # Function calls another function
    INHERITS = "inherits"         # Class inherits from another class
    USES = "uses"                 # Symbol uses another symbol
    CONTAINS = "contains"         # Module/class contains function/method
    DECORATES = "decorates"       # Decorator applied to function/class


class NodeType(Enum):
    """Types of nodes in the graph"""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"


@dataclass
class GraphNode:
    """Represents a node in the dependency graph"""
    id: str                       # Unique identifier (e.g., "module.Class.method")
    name: str                     # Short name
    node_type: NodeType
    file_path: str
    line_number: int = 0
    signature: str = ""
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "file": self.file_path,
            "line": self.line_number,
            "signature": self.signature,
            "docstring": self.docstring,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType(data["type"]),
            file_path=data["file"],
            line_number=data.get("line", 0),
            signature=data.get("signature", ""),
            docstring=data.get("docstring"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GraphEdge:
    """Represents an edge in the dependency graph"""
    source: str                   # Source node ID
    target: str                   # Target node ID
    edge_type: EdgeType
    weight: float = 1.0           # Edge weight (for ranking)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=EdgeType(data["type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


class DependencyGraph:
    """
    Manages the code dependency graph using NetworkX.
    Tracks imports, calls, inheritance, and usage relationships.
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path
        if self.persist_path is None:
            cache_path = get_cache_path()
            self.persist_path = cache_path / "dependency_graph.json"
        
        # Use directed multigraph to allow multiple edge types between nodes
        self.graph = nx.DiGraph()
        
        # Node metadata storage
        self.nodes: Dict[str, GraphNode] = {}
        
        # Track indexed files
        self.indexed_files: Dict[str, float] = {}  # file -> timestamp
        
        # Load existing graph if available
        self._load()
    
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph"""
        if node.id in self.nodes:
            # Update existing node
            self.nodes[node.id] = node
        else:
            self.nodes[node.id] = node
        
        self.graph.add_node(
            node.id,
            **node.to_dict()
        )
        return True
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph"""
        # Ensure both nodes exist
        if edge.source not in self.graph:
            return False
        
        # Target might be external (not in our codebase)
        if edge.target not in self.graph:
            # Create placeholder node for external references
            self.graph.add_node(
                edge.target,
                id=edge.target,
                name=edge.target.split(".")[-1],
                type="external",
                file="",
                line=0,
            )
        
        self.graph.add_edge(
            edge.source,
            edge.target,
            **edge.to_dict()
        )
        return True
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_dependencies(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[str, EdgeType]]:
        """
        Get all nodes that this node depends on (outgoing edges).
        
        Args:
            node_id: Source node ID
            edge_types: Filter by edge types (None = all)
            
        Returns:
            List of (target_id, edge_type) tuples
        """
        if node_id not in self.graph:
            return []
        
        dependencies = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            edge_type = EdgeType(data.get("type", "uses"))
            if edge_types is None or edge_type in edge_types:
                dependencies.append((target, edge_type))
        
        return dependencies
    
    def get_dependents(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[Tuple[str, EdgeType]]:
        """
        Get all nodes that depend on this node (incoming edges).
        
        Args:
            node_id: Target node ID
            edge_types: Filter by edge types (None = all)
            
        Returns:
            List of (source_id, edge_type) tuples
        """
        if node_id not in self.graph:
            return []
        
        dependents = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            edge_type = EdgeType(data.get("type", "uses"))
            if edge_types is None or edge_type in edge_types:
                dependents.append((source, edge_type))
        
        return dependents
    
    def get_related_nodes(
        self,
        node_id: str,
        max_depth: int = 2,
        max_nodes: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get related nodes using BFS traversal.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to return
            
        Returns:
            List of related nodes with distance and relationship
        """
        if node_id not in self.graph:
            return []
        
        related = []
        visited = {node_id}
        queue = [(node_id, 0, "self")]
        
        while queue and len(related) < max_nodes:
            current, depth, relationship = queue.pop(0)
            
            if depth > 0:  # Skip the starting node
                node = self.nodes.get(current)
                if node:
                    related.append({
                        "id": current,
                        "name": node.name,
                        "type": node.node_type.value,
                        "file": node.file_path,
                        "distance": depth,
                        "relationship": relationship,
                    })
            
            if depth >= max_depth:
                continue
            
            # Add neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge_data = self.graph.get_edge_data(current, neighbor)
                    rel = edge_data.get("type", "related") if edge_data else "related"
                    queue.append((neighbor, depth + 1, rel))
            
            # Also check incoming edges
            for predecessor in self.graph.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    edge_data = self.graph.get_edge_data(predecessor, current)
                    rel = f"used_by_{edge_data.get('type', 'related')}" if edge_data else "used_by"
                    queue.append((predecessor, depth + 1, rel))
        
        return related
    
    def find_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_module_graph(self, file_path: str) -> Dict[str, Any]:
        """Get subgraph for a specific file"""
        file_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.file_path == file_path
        ]
        
        subgraph = self.graph.subgraph(file_nodes)
        
        return {
            "nodes": [self.nodes[n].to_dict() for n in file_nodes if n in self.nodes],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": d.get("type", "unknown"),
                }
                for u, v, d in subgraph.edges(data=True)
            ],
        }
    
    def remove_file(self, file_path: str) -> int:
        """Remove all nodes and edges for a file (for re-indexing)"""
        nodes_to_remove = [
            node_id for node_id, node in self.nodes.items()
            if node.file_path == file_path
        ]
        
        for node_id in nodes_to_remove:
            self.graph.remove_node(node_id)
            del self.nodes[node_id]
        
        if file_path in self.indexed_files:
            del self.indexed_files[file_path]
        
        return len(nodes_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        node_types = {}
        edge_types = {}
        
        for node in self.nodes.values():
            t = node.node_type.value
            node_types[t] = node_types.get(t, 0) + 1
        
        for _, _, data in self.graph.edges(data=True):
            t = data.get("type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "indexed_files": len(self.indexed_files),
            "node_types": node_types,
            "edge_types": edge_types,
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }
    
    def search_nodes(
        self,
        query: str,
        node_types: Optional[List[NodeType]] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """Search nodes by name (simple substring match)"""
        query_lower = query.lower()
        results = []
        
        for node in self.nodes.values():
            if node_types and node.node_type not in node_types:
                continue
            
            # Check name and signature
            if query_lower in node.name.lower() or query_lower in node.signature.lower():
                results.append(node)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def mark_file_indexed(self, file_path: str):
        """Mark a file as indexed"""
        self.indexed_files[file_path] = time.time()
    
    def is_file_indexed(self, file_path: str, max_age: float = 3600) -> bool:
        """Check if a file is indexed (within max_age seconds)"""
        if file_path not in self.indexed_files:
            return False
        
        age = time.time() - self.indexed_files[file_path]
        return age < max_age
    
    def save(self):
        """Persist graph to disk"""
        data = {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **d
                }
                for u, v, d in self.graph.edges(data=True)
            ],
            "indexed_files": self.indexed_files,
            "saved_at": time.time(),
        }
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load graph from disk"""
        if not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            # Restore nodes
            for node_id, node_data in data.get("nodes", {}).items():
                node = GraphNode.from_dict(node_data)
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **node_data)
            
            # Restore edges
            for edge_data in data.get("edges", []):
                self.graph.add_edge(
                    edge_data["source"],
                    edge_data["target"],
                    **{k: v for k, v in edge_data.items() if k not in ("source", "target")}
                )
            
            # Restore indexed files
            self.indexed_files = data.get("indexed_files", {})
            
        except Exception as e:
            print(f"Error loading graph: {e}")
            # Start fresh
            self.graph = nx.DiGraph()
            self.nodes = {}
            self.indexed_files = {}
    
    def clear(self):
        """Clear all graph data"""
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.indexed_files = {}
        
        if self.persist_path.exists():
            self.persist_path.unlink()


# Singleton instance
_graph: Optional[DependencyGraph] = None


def get_dependency_graph() -> DependencyGraph:
    """Get or create the global dependency graph instance"""
    global _graph
    if _graph is None:
        _graph = DependencyGraph()
    return _graph



