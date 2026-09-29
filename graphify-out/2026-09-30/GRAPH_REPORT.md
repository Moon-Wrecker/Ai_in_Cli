# Graph Report - Ai_in_Cli  (2026-09-30)

## Corpus Check
- 34 files · ~226,366 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1006 nodes · 1733 edges · 60 communities (51 shown, 5 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 150 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `01dc37d0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- get_settings
- IndexManager
- FunctionRegistry
- AI CLI v2.0 - Complete Documentation
- parsers.py
- CodeTools
- AI in CLI - Application Capabilities
- FolderTools
- PythonASTParser
- ASTIndexer
- 🤖 AI CLI v2.0
- ChromaVectorStore
- SemanticIndexer
- main.py
- CallGraphVisitor
- SearchTools
- indexing/__init__.py
- core/__init__.py
- DependencyGraphBuilder
- DependencyGraph
- .search
- terminal_tools.py
- CSSParser
- .__init__
- FileTools
- KeywordSearcher
- graph_store.py
- ConversationMemory
- Message
- GraphNode
- tools/__init__.py
- conversation.py
- Any
- _MockPosthog
- OpenAIEmbeddings
- 🤖 AI CLI v2.0
- MockPosthog
- ContextManager
- dependency_graph.py
- TicTacToe
- chroma_store.py
- SemanticIndexStats
- 🛠️ Available Tools (25)
- 💬 Example Requests
- ._trigger_index
- CodeSymbol
- 🚀 Quick Start
- 🔄 Smart Indexing
- 🐛 Troubleshooting
- ⚡ Performance Notes
- 📖 Usage Guide
- .get_file_info
- search_symbol
- 🔒 Security
- 🔍 Search System
- .index_manager

## God Nodes (most connected - your core abstractions)
1. `ASTIndexer` - 39 edges
2. `DependencyGraphBuilder` - 29 edges
3. `DependencyGraph` - 28 edges
4. `SearchTools` - 28 edges
5. `PythonASTParser` - 28 edges
6. `FileTools` - 26 edges
7. `HybridRetriever` - 25 edges
8. `CodeTools` - 25 edges
9. `get_settings()` - 23 edges
10. `FunctionRegistry` - 23 edges

## Surprising Connections (you probably didn't know these)
- `ChromaVectorStore` --uses--> `Settings`  [INFERRED]
  ai_cli/storage/chroma_store.py → ai_cli/config.py
- `get_config()` --calls--> `get_settings()`  [INFERRED]
  ai_cli/main.py → ai_cli/config.py
- `get_config()` --calls--> `get_sandbox_path()`  [INFERRED]
  ai_cli/main.py → ai_cli/config.py
- `AIAgent` --uses--> `ConversationMemory`  [INFERRED]
  ai_cli/core/agent.py → ai_cli/core/conversation.py
- `AIAgent` --uses--> `FunctionRegistry`  [INFERRED]
  ai_cli/core/agent.py → ai_cli/core/function_registry.py

## Import Cycles
- None detected.

## Communities (60 total, 5 thin omitted)

### Community 0 - "get_settings"
Cohesion: 0.05
Nodes (40): Config, get_cache_path(), get_chroma_path(), get_sandbox_path(), get_settings(), Path, Configuration for AI CLI Assistant Central configuration management using…, Get absolute path to cache directory (+32 more)

### Community 1 - "IndexManager"
Cohesion: 0.07
Nodes (32): FileState, get_index_manager(), IndexManager, IndexState, on_file_created(), on_file_deleted(), on_file_modified(), Any (+24 more)

### Community 2 - "FunctionRegistry"
Cohesion: 0.07
Nodes (28): FunctionRegistry, Any, Definition of a tool parameter, Register folder operation tools, Complete definition of a tool, Register code editing tools, Convert to OpenAI function schema, Register terminal/command tools (+20 more)

### Community 3 - "AI CLI v2.0 - Complete Documentation"
Cohesion: 0.04
Nodes (46): 1. Agent (`core/agent.py`), 1. AST Indexer (`indexing/ast_indexer.py`), 2. Function Registry (`core/function_registry.py`), 2. Semantic Indexer (`indexing/semantic_indexer.py`), 3. Conversation Memory (`core/conversation.py`), 3. Dependency Graph (`indexing/dependency_graph.py`), 4. Hybrid Retriever (`indexing/hybrid_retriever.py`), Agent Loop Flow (+38 more)

### Community 4 - "parsers.py"
Cohesion: 0.09
Nodes (30): Utility modules for AI CLI, CodeChunker, extract_symbols(), get_file_structure(), get_language(), MultiLanguageParser, parse_file(), parse_python_file() (+22 more)

### Community 5 - "CodeTools"
Cohesion: 0.08
Nodes (26): analyze_code(), CodeTools, delete_lines(), find_and_replace(), insert_lines(), Any, Path, Code Tools - Smart code editing with line-level precision Provides intelligent… (+18 more)

### Community 6 - "AI in CLI - Application Capabilities"
Cohesion: 0.06
Nodes (33): 1. **Natural Language Command Processing**, 2. **File System Operations**, 3. **Smart Code Editing**, 4. **RAG-Powered Workspace Intelligence**, 5. **Safe Terminal Command Execution**, 6. **Multiple Interface Modes**, 7. **Cross-Platform Compatibility**, 8. **Security & Sandboxing** (+25 more)

### Community 7 - "FolderTools"
Cohesion: 0.09
Nodes (22): create_directory(), delete_directory(), find_files(), FolderTools, list_directory(), Any, Path, Folder Tools - Directory operations All operations are sandboxed to the sandbox… (+14 more)

### Community 8 - "PythonASTParser"
Cohesion: 0.10
Nodes (20): AST, AsyncFunctionDef, ClassDef, FunctionDef, PythonASTParser, Advanced Python AST parser for deep code analysis. Extracts symbols,…, Count non-empty, non-comment lines, Extract all import statements (+12 more)

### Community 9 - "ASTIndexer"
Cohesion: 0.11
Nodes (16): ASTIndexer, Any, Path, Index all supported files in the sandbox workspace. Supports: Python,…, Index a single file (multi-language support). Supports: Python, JavaScript,…, Find all indexable files in sandbox (multi-language), Find all Python files in sandbox (backwards compatibility), Generate module ID from file path (+8 more)

### Community 10 - "🤖 AI CLI v2.0"
Cohesion: 0.07
Nodes (28): A Claude Code-level AI Terminal Assistant, 🎮 Advanced Capabilities, 🤖 AI CLI v2.0, ✅ Allowed, 🏗️ Architecture, ❌ Blocked, Blocked Commands, 🙏 Built With (+20 more)

### Community 11 - "ChromaVectorStore"
Cohesion: 0.10
Nodes (16): ChromaVectorStore, Any, Path, ChromaDB-based vector store for code chunks. Supports both in-memory and…, Initialize ChromaDB client, Initialize or get collection, Add code chunks to the vector store. Args: chunks: List of chunk dictionaries…, Semantic search for relevant code chunks. Args: query: Search query n_results:… (+8 more)

### Community 12 - "SemanticIndexer"
Cohesion: 0.12
Nodes (15): Any, Path, Index a single file with semantic embeddings. Returns: Dictionary with indexing…, Find all files that can be indexed, Create simple line-based chunks for non-Python files, Enrich chunks with additional context for better retrieval. Adds file context,…, Get context information about a file, Build enriched content string for embedding (+7 more)

### Community 13 - "main.py"
Cohesion: 0.15
Nodes (21): get_config(), handle_special_command(), main(), print_banner(), print_help_info(), print_workspace_info(), Handle special commands starting with / Returns: True if command was handled,…, Run incremental indexing on startup if needed (+13 more)

### Community 14 - "CallGraphVisitor"
Cohesion: 0.11
Nodes (14): CallGraphVisitor, AST, AsyncFunctionDef, ClassDef, FunctionDef, Common logic for function visiting, Record function calls, Extract the name of a called function (+6 more)

### Community 15 - "SearchTools"
Cohesion: 0.11
Nodes (15): index_workspace(), Any, Get code context around a specific line. Args: file_path: Path to file…, Search for definitions (functions, classes, etc.). Args: name: Name to search…, Provides advanced search capabilities using hybrid retrieval. Combines semantic…, Get call graph for a function. Args: function_name: Function name Returns: Dict…, Get an overview of the indexed workspace. Returns: Dict with workspace…, Index the workspace for search. Args: force: If True, re-index everything… (+7 more)

### Community 16 - "indexing/__init__.py"
Cohesion: 0.15
Nodes (15): index_sandbox(), IndexStats, Multi-Language AST Indexer - Symbol-aware code indexing Supports Python (AST),…, Statistics from indexing operation, Index the entire sandbox workspace, hybrid_search(), HybridRetriever, Hybrid Retriever - Combines semantic, keyword, and graph-based search Uses… (+7 more)

### Community 17 - "core/__init__.py"
Cohesion: 0.14
Nodes (15): AgentResponse, AIAgent, chat(), get_agent(), Any, AI Agent - OpenAI Function Calling Loop Core agent implementation without…, Process a user message and return a response. Implements the main agent loop…, Process a user message with streaming response. Yields content chunks as they… (+7 more)

### Community 18 - "DependencyGraphBuilder"
Cohesion: 0.16
Nodes (12): DependencyGraphBuilder, Path, Builds comprehensive dependency graphs for Python codebases. Analyzes imports,…, Build dependency graph for all Python files in sandbox. Args: force: If True,…, Add a file's symbols and imports to the graph, Add a symbol to the graph, Add import edge to graph, Analyze function calls in a file (+4 more)

### Community 19 - "DependencyGraph"
Cohesion: 0.10
Nodes (10): DependencyGraph, Path, Manages the code dependency graph using NetworkX. Tracks imports, calls,…, Add an edge to the graph, Get all nodes that this node depends on (outgoing edges). Args: node_id: Source…, Find shortest path between two nodes, Remove all nodes and edges for a file (for re-indexing), Mark a file as indexed (+2 more)

### Community 20 - ".search"
Cohesion: 0.12
Nodes (11): HybridSearchResult, Any, Result from hybrid search with combined scoring, Perform hybrid search combining multiple retrieval methods. Args: query: Search…, Get semantic search results, Get keyword search results, Get graph-based search results, Combine rankings using Reciprocal Rank Fusion. RRF score = sum(1 / (k + rank))… (+3 more)

### Community 21 - "terminal_tools.py"
Cohesion: 0.11
Nodes (18): check_command_safety(), check_display_available(), detect_gui_imports(), execute_command(), get_gui_environment(), get_system_info(), Terminal Tools - Safe command execution Provides sandboxed terminal command…, Run Python code or script with full GUI support. Args: code: Python code to… (+10 more)

### Community 22 - "CSSParser"
Cohesion: 0.15
Nodes (9): CSSParser, Regex-based CSS parser for selector and variable extraction. Supports CSS,…, Parse a CSS/SCSS/LESS file, Extract @import statements, Parse a Python file and extract all symbols and metadata, Parse code string directly, Extract __all__ definition if present, Parse a JavaScript/TypeScript file (+1 more)

### Community 23 - ".__init__"
Cohesion: 0.12
Nodes (10): HTMLParser, ImportInfo, JavaScriptParser, Extract linked resources (CSS, JS), Represents an import statement, Regex-based JavaScript/TypeScript parser for symbol extraction. Extracts…, Extract import statements, Extract all symbols from JavaScript/TypeScript code (+2 more)

### Community 24 - "FileTools"
Cohesion: 0.16
Nodes (11): create_file(), FileTools, Any, Read specific lines from a file with line numbers. Args: filepath: Path to file…, Provides file CRUD operations with security sandboxing. All operations are…, Write/overwrite a file with content. Args: filepath: Path to file content: New…, Append content to an existing file. Args: filepath: Path to file content:…, Copy a file to a new location. Args: source: Source file path destination:… (+3 more)

### Community 25 - "KeywordSearcher"
Cohesion: 0.18
Nodes (9): KeywordSearcher, Path, Get all searchable files with safety limits, Tokenize text into search terms, Calculate BM25-like relevance score., Lazy load keyword searcher, Simple keyword-based search using regex matching. Supports BM25-like scoring…, Search for keyword matches in files. Returns: List of (file_path, start_line,… (+1 more)

### Community 26 - "graph_store.py"
Cohesion: 0.17
Nodes (11): EdgeType, get_dependency_graph(), NodeType, Enum, Graph Store for dependency and call graphs Uses NetworkX for graph operations…, Get all nodes that depend on this node (incoming edges). Args: node_id: Target…, Types of edges in the dependency graph, Types of nodes in the graph (+3 more)

### Community 27 - "ConversationMemory"
Cohesion: 0.19
Nodes (8): ConversationMemory, Add a tool result message, Get a summary of recent conversation context. Useful for providing context…, Clear conversation history, Trim messages to window size, Save conversation to disk, Manages conversation history with windowing and persistence., Add an assistant message

### Community 28 - "Message"
Cohesion: 0.15
Nodes (9): Message, Any, Get messages in OpenAI format. Args: include_system: Include system message…, Load a previous conversation session, Represents a conversation message, Get conversation statistics, Convert to OpenAI message format, Convert to OpenAI API message format (+1 more)

### Community 29 - "GraphNode"
Cohesion: 0.14
Nodes (7): GraphNode, Any, Add a node to the graph, Get related nodes using BFS traversal. Args: node_id: Starting node ID…, Get subgraph for a specific file, Search nodes by name (simple substring match), Represents a node in the dependency graph

### Community 30 - "tools/__init__.py"
Cohesion: 0.23
Nodes (10): delete_file(), File Tools - CRUD operations for files All operations are sandboxed to the…, Read specific lines from a file, Write content to a file, read_file(), read_file_lines(), write_file(), Tools modules for AI CLI (+2 more)

### Community 31 - "conversation.py"
Cohesion: 0.20
Nodes (7): Clear conversation memory, Set the system message for the agent, get_context_manager(), get_memory(), Conversation Memory Management Handles conversation history and context, Get or create conversation memory, Get or create context manager

### Community 32 - "Any"
Cohesion: 0.24
Nodes (5): Any, Get import tree for a module, Get callers tree for a symbol, Build dependency tree, Build reverse dependency tree (who uses this)

### Community 34 - "OpenAIEmbeddings"
Cohesion: 0.27
Nodes (5): OpenAIEmbeddings, Embed a batch of texts, Generate cache key for text, OpenAI embeddings wrapper for ChromaDB. Uses text-embedding-3-small by default…, Embed multiple documents

### Community 35 - "🤖 AI CLI v2.0"
Cohesion: 0.22
Nodes (8): 🤖 AI CLI v2.0, ⚙️ Configuration, 🙏 Credits, Environment Variables, ✨ Features, 📝 License, 📁 Project Structure, 📊 Statistics

### Community 37 - "ContextManager"
Cohesion: 0.29
Nodes (5): ContextManager, Manages additional context for the conversation. Provides workspace awareness…, Get current workspace context, Build workspace context string, Get context for a specific file

### Community 38 - "dependency_graph.py"
Cohesion: 0.29
Nodes (5): build_dependency_graph(), DependencyStats, Dependency Graph Builder - Analyzes import and call relationships Creates a…, Statistics from dependency analysis, Build dependency graph for sandbox

### Community 40 - "chroma_store.py"
Cohesion: 0.33
Nodes (4): Lazy load vector store, get_vector_store(), ChromaDB Vector Store for semantic search Provides in-memory and persistent…, Get or create the global vector store instance

### Community 41 - "SemanticIndexStats"
Cohesion: 0.33
Nodes (4): index_semantic(), Statistics from semantic indexing, Index the sandbox with semantic embeddings, SemanticIndexStats

### Community 42 - "🛠️ Available Tools (25)"
Cohesion: 0.33
Nodes (6): 🛠️ Available Tools (25), Code Tools, File Tools, Folder Tools, Search Tools, Terminal Tools

### Community 43 - "💬 Example Requests"
Cohesion: 0.33
Nodes (6): Code Editing, Directory Operations, 💬 Example Requests, File Operations, Search & Analysis, Terminal Commands

### Community 45 - "._trigger_index"
Cohesion: 0.33
Nodes (3): Create a new file with content. Args: filepath: Path for new file content: File…, Delete a file. Args: filepath: Path to file Returns: Dict with deletion result, Trigger indexing for a file after an operation

### Community 46 - "CodeSymbol"
Cohesion: 0.33
Nodes (4): CodeSymbol, Extract symbols from HTML, Represents a code symbol with full metadata, Convert to dictionary for serialization

### Community 47 - "🚀 Quick Start"
Cohesion: 0.40
Nodes (5): 1. Prerequisites, 2. Installation, 3. Configuration, 4. Run, 🚀 Quick Start

### Community 48 - "🔄 Smart Indexing"
Cohesion: 0.40
Nodes (5): Check Index Status, How Files Are Tracked, How It Works, 🔄 Smart Indexing, What Gets Indexed

### Community 49 - "🐛 Troubleshooting"
Cohesion: 0.40
Nodes (5): Command blocked, "File not found" errors, "OPENAI_API_KEY not found", Search returns no results, 🐛 Troubleshooting

### Community 50 - "⚡ Performance Notes"
Cohesion: 0.40
Nodes (5): File Limits, Ignored Directories, Lazy Loading, ⚡ Performance Notes, Resource-Intensive Operations

### Community 51 - "📖 Usage Guide"
Cohesion: 0.50
Nodes (4): CLI Commands, Interactive Mode (Recommended), Slash Commands, 📖 Usage Guide

### Community 53 - "search_symbol"
Cohesion: 0.50
Nodes (3): Get code related to a file or symbol. Args: file_path: Path to file…, Search for a code symbol, search_symbol()

### Community 54 - "🔒 Security"
Cohesion: 0.67
Nodes (3): Command Safety, Sandboxing, 🔒 Security

### Community 55 - "🔍 Search System"
Cohesion: 0.67
Nodes (3): Hybrid Retrieval, Indexing, 🔍 Search System

## Knowledge Gaps
- **120 isolated node(s):** `Config`, `A Claude Code-level AI Terminal Assistant`, `🏗️ Architecture`, `🧠 Intelligent Code Understanding`, `🎮 Advanced Capabilities` (+115 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 519 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SearchTools` connect `SearchTools` to `get_settings`, `FunctionRegistry`, `ASTIndexer`, `SemanticIndexer`, `main.py`, `indexing/__init__.py`, `DependencyGraphBuilder`, `search_symbol`, `graph_store.py`, `tools/__init__.py`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `ASTIndexer` connect `ASTIndexer` to `IndexManager`, `parsers.py`, `PythonASTParser`, `SemanticIndexer`, `CodeSymbol`, `SearchTools`, `indexing/__init__.py`, `DependencyGraphBuilder`, `DependencyGraph`, `CSSParser`, `.__init__`, `graph_store.py`, `GraphNode`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `FunctionRegistry` connect `FunctionRegistry` to `CodeTools`, `FolderTools`, `SearchTools`, `core/__init__.py`, `FileTools`, `conversation.py`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `ASTIndexer` (e.g. with `DependencyGraph` and `EdgeType`) actually correct?**
  _`ASTIndexer` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `DependencyGraphBuilder` (e.g. with `DependencyGraph` and `EdgeType`) actually correct?**
  _`DependencyGraphBuilder` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `DependencyGraph` (e.g. with `ASTIndexer` and `DependencyGraphBuilder`) actually correct?**
  _`DependencyGraph` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `SearchTools` (e.g. with `FunctionRegistry` and `._register_search_tools()`) actually correct?**
  _`SearchTools` has 10 INFERRED edges - model-reasoned connections that need verification._