# AI CLI v2.0 - Complete Documentation

## Overview

**AI CLI v2.0** is a Claude Code / Codex-level terminal AI assistant built **without LangChain**. It uses direct OpenAI API calls with function calling for a cleaner, more efficient implementation.

### Key Features

- 🤖 **Direct OpenAI Integration** - No LangChain overhead
- 🔍 **Hybrid Retrieval** - Semantic + Keyword + Graph search with RRF fusion
- 🌳 **Symbol-Aware AST Indexing** - Deep Python code understanding
- 📊 **Dependency Graph** - Import and call relationship tracking
- 🔒 **Sandboxed Operations** - All file operations restricted to sandbox
- 💾 **ChromaDB Vector Store** - Persistent semantic search
- 🎨 **Beautiful Rich UI** - Interactive terminal interface

---

## Architecture

```
ai_cli/
├── main.py                      # Entry point with Rich UI
├── config.py                    # Configuration management
├── requirements.txt             # Dependencies
│
├── core/                        # Core agent components
│   ├── agent.py                 # OpenAI function calling loop
│   ├── conversation.py          # Memory management
│   └── function_registry.py     # Tool definitions & schemas
│
├── indexing/                    # Code indexing system
│   ├── ast_indexer.py           # Python AST parsing
│   ├── semantic_indexer.py      # OpenAI embeddings
│   ├── dependency_graph.py      # Import/call graph builder
│   └── hybrid_retriever.py      # RRF fusion search
│
├── tools/                       # Tool implementations
│   ├── file_tools.py            # File CRUD operations
│   ├── folder_tools.py          # Directory operations
│   ├── code_tools.py            # Smart code editing
│   ├── terminal_tools.py        # Safe command execution
│   └── search_tools.py          # Hybrid search interface
│
├── storage/                     # Data persistence
│   ├── chroma_store.py          # Vector database
│   └── graph_store.py           # Dependency graph persistence
│
├── utils/                       # Utilities
│   ├── security.py              # Sandboxing & validation
│   └── parsers.py               # Python code parsers
│
└── sandbox/                     # Sandboxed workspace
```

---

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

```bash
cd ai_cli

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Run

```bash
# Interactive mode
python main.py

# Single command
python main.py chat "List all Python files"

# Index workspace
python main.py index

# Search
python main.py search "authentication logic"
```

---

## Core Components

### 1. Agent (`core/agent.py`)

The heart of the system - implements OpenAI function calling loop.

```python
class AIAgent:
    def chat(self, user_message: str) -> AgentResponse:
        """Main agent loop with tool execution"""
        
    def chat_stream(self, user_message: str) -> Generator[str, None, AgentResponse]:
        """Streaming response for better UX"""
```

**How it works:**
1. User sends message
2. Message added to conversation memory
3. OpenAI API called with tool schemas
4. If tool calls returned:
   - Execute each tool
   - Add results to memory
   - Loop back to step 3
5. Return final response

### 2. Function Registry (`core/function_registry.py`)

Manages all available tools with OpenAI-compatible schemas.

```python
# 25 tools registered:
- read_file, read_file_lines, create_file, write_file, delete_file
- list_directory, create_directory, delete_directory, find_files
- insert_lines, replace_lines, delete_lines, find_and_replace
- execute_command, run_python_code, run_python_script
- search_codebase, search_symbol, index_workspace
- analyze_code, apply_edit, get_file_info, get_directory_info
- get_system_info, get_workspace_overview
```

### 3. Conversation Memory (`core/conversation.py`)

Handles conversation history with windowing.

```python
class ConversationMemory:
    def add_user_message(self, content: str)
    def add_assistant_message(self, content: str, tool_calls=None)
    def add_tool_result(self, tool_call_id, tool_name, result)
    def get_messages(self) -> List[Dict]  # OpenAI format
```

---

## Indexing System

### 1. AST Indexer (`indexing/ast_indexer.py`)

Deep Python code understanding through AST analysis.

**Extracts:**
- Classes with inheritance hierarchy
- Functions/methods with signatures
- Docstrings
- Decorators
- Parameters and return types
- Cyclomatic complexity estimates

```python
@dataclass
class CodeSymbol:
    name: str
    symbol_type: SymbolType  # CLASS, FUNCTION, METHOD, etc.
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str]
    parameters: List[str]
    return_type: Optional[str]
    base_classes: List[str]
    complexity: int
```

### 2. Semantic Indexer (`indexing/semantic_indexer.py`)

Creates vector embeddings using OpenAI's `text-embedding-3-small`.

**Process:**
1. Code split into semantic chunks (respecting function/class boundaries)
2. Chunks enriched with context (file path, symbol info)
3. Embeddings generated via OpenAI API
4. Stored in ChromaDB for similarity search

### 3. Dependency Graph (`indexing/dependency_graph.py`)

Builds comprehensive code relationship graph.

**Tracks:**
- **Import edges**: Module imports
- **Call edges**: Function calls
- **Inheritance edges**: Class inheritance
- **Containment edges**: Module contains class/function

**Features:**
- Circular dependency detection
- Caller/callee tree traversal
- Symbol search by name

### 4. Hybrid Retriever (`indexing/hybrid_retriever.py`)

Combines three search methods using **Reciprocal Rank Fusion (RRF)**.

```python
RRF_score = Σ (weight / (k + rank))

Default weights:
- Semantic: 0.4
- Keyword: 0.3  
- Graph: 0.3
```

**Search types:**
1. **Semantic**: Vector similarity via ChromaDB
2. **Keyword**: BM25-like term matching
3. **Graph**: Symbol relationship traversal

---

## Tools

### File Operations (`tools/file_tools.py`)

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with encoding detection |
| `read_file_lines` | Read specific line range with line numbers |
| `create_file` | Create new file with content |
| `write_file` | Overwrite file content |
| `delete_file` | Delete a file |
| `get_file_info` | Get file metadata (size, dates, permissions) |

### Folder Operations (`tools/folder_tools.py`)

| Tool | Description |
|------|-------------|
| `list_directory` | List directory contents (flat or recursive tree) |
| `create_directory` | Create directory (with parents) |
| `delete_directory` | Delete directory (optional recursive) |
| `find_files` | Find files matching glob pattern |
| `get_directory_info` | Get directory statistics |

### Code Editing (`tools/code_tools.py`)

| Tool | Description |
|------|-------------|
| `insert_lines` | Insert content at specific line |
| `replace_lines` | Replace line range with new content |
| `delete_lines` | Delete line range |
| `find_and_replace` | Find/replace with regex support |
| `apply_edit` | Precise edit by finding exact content |
| `analyze_code` | Analyze Python code structure |

**Features:**
- Automatic backups before edits
- Similar content detection for failed edits
- Diff generation

### Terminal Commands (`tools/terminal_tools.py`)

| Tool | Description |
|------|-------------|
| `execute_command` | Run shell command (validated for safety) |
| `run_python_code` | Execute Python code directly |
| `run_python_script` | Run Python script file |
| `get_system_info` | Get OS/Python/shell information |

**Safety features:**
- Command whitelist/blacklist
- Blocked patterns (rm -rf /, etc.)
- Restricted environment
- Timeout enforcement

### Search Tools (`tools/search_tools.py`)

| Tool | Description |
|------|-------------|
| `search_codebase` | Hybrid search with RRF fusion |
| `search_symbol` | Find symbol by name |
| `get_workspace_overview` | Workspace statistics |
| `index_workspace` | Index for semantic search |

---

## Storage

### ChromaDB Vector Store (`storage/chroma_store.py`)

Persistent vector database for semantic search.

```python
class ChromaVectorStore:
    def add_chunks(self, chunks: List[Dict]) -> int
    def search(self, query: str, n_results: int) -> List[SearchResult]
    def delete_file_chunks(self, file_path: str) -> int
    def get_stats(self) -> Dict
```

**Features:**
- Cosine similarity search
- File-based chunk management
- Statistics and monitoring

### Graph Store (`storage/graph_store.py`)

NetworkX-based dependency graph with JSON persistence.

```python
class DependencyGraph:
    def add_node(self, node: GraphNode) -> bool
    def add_edge(self, edge: GraphEdge) -> bool
    def get_dependencies(self, node_id: str) -> List[Tuple]
    def get_dependents(self, node_id: str) -> List[Tuple]
    def get_related_nodes(self, node_id: str, max_depth: int) -> List[Dict]
    def find_path(self, source: str, target: str) -> List[str]
```

---

## Security

### Path Validation (`utils/security.py`)

All file operations are sandboxed to the `sandbox/` directory.

```python
class PathValidator:
    def resolve_path(self, user_path: str) -> Path:
        """Resolve and validate path within sandbox"""
        # Prevents:
        # - Directory traversal (../)
        # - Absolute paths outside sandbox
        # - Symlink escapes
```

### Command Validation

```python
class CommandValidator:
    # Blocked patterns:
    blocked_commands = [
        "rm -rf /", "mkfs", "dd if=", "chmod -R 777 /",
        "shutdown", "reboot", ":(){:|:&};:", ...
    ]
    
    def validate_command(self, command: str) -> Tuple[bool, str, str]:
        """Returns (is_safe, risk_level, message)"""
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Model Configuration
OPENAI_MODEL=gpt-4o  # or gpt-4-turbo, gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.1

# Agent
MAX_ITERATIONS=15
MAX_TOKENS=4096
MEMORY_WINDOW_SIZE=20

# Paths
SANDBOX_DIR=sandbox
CACHE_DIR=.cache
CHROMA_PERSIST_DIR=.chroma_db

# Indexing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=10

# Hybrid Retrieval Weights
SEMANTIC_WEIGHT=0.4
KEYWORD_WEIGHT=0.3
GRAPH_WEIGHT=0.3
```

---

## Usage Examples

### Interactive Mode

```bash
python main.py
```

Commands:
- `/help` - Show commands
- `/clear` - Clear conversation
- `/index` - Index workspace
- `/stats` - Show statistics
- `/tools` - List available tools
- `/workspace` - Show workspace info
- `/exit` - Exit

### Natural Language Examples

```
> Create a file called hello.py with a greeting function

> List all Python files in the workspace

> Search for "database connection" in the codebase

> Read lines 10-50 of config.py

> Replace "old_function" with "new_function" in utils.py

> Run the tests in test_main.py

> Analyze the code structure of app.py

> Find all functions that call "validate_user"
```

---

## Comparison with LangChain Version

| Aspect | v1 (LangChain) | v2 (Direct OpenAI) |
|--------|---------------|-------------------|
| Dependencies | 15+ packages | 8 packages |
| Code complexity | High (abstractions) | Low (direct) |
| Debug-ability | Hard | Easy |
| Performance | Overhead | Minimal |
| Customization | Limited by framework | Full control |
| Tool schemas | LangChain format | OpenAI native |
| Streaming | Complex | Simple |

---

## Technical Details

### Agent Loop Flow

```
┌─────────────────────────────────────────────────────────┐
│                     User Message                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Add to Conversation Memory                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           OpenAI API Call with Tool Schemas             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Tool Calls?  │
                   └──────────────┘
                    │           │
              Yes   │           │  No
                    ▼           ▼
         ┌──────────────┐  ┌──────────────┐
         │Execute Tools │  │Final Response│
         │Add Results   │  └──────────────┘
         └──────────────┘
                    │
                    └──────────── Loop back to API call
```

### Hybrid Retrieval Flow

```
                    Query
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Semantic │ │ Keyword  │ │  Graph   │
    │  Search  │ │  Search  │ │  Search  │
    └──────────┘ └──────────┘ └──────────┘
          │           │           │
          └───────────┼───────────┘
                      ▼
             ┌─────────────────┐
             │   RRF Fusion    │
             │ (Rank Combine)  │
             └─────────────────┘
                      │
                      ▼
              Combined Results
```

---

## Performance Tips

1. **Index workspace first**: Run `/index` before searching
2. **Use specific queries**: More specific = better results
3. **Leverage tools**: Let the AI use tools instead of asking for information
4. **Clear memory**: Use `/clear` for new topics
5. **Check stats**: Use `/stats` to monitor token usage

---

## Troubleshooting

### "OPENAI_API_KEY not found"
Create `.env` file with your API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### "File not found" errors
All paths are relative to `sandbox/` directory. Use:
```
read_file test.py  # Not /absolute/path/test.py
```

### Search returns no results
Run indexing first:
```bash
python main.py index
```

### Command blocked
Some commands are blocked for safety. Check with:
```
Is "your_command" safe to execute?
```

---

## License

MIT License

---

## Version History

- **v2.0.0** - Complete rewrite without LangChain
  - Direct OpenAI function calling
  - Hybrid retrieval with RRF fusion
  - Symbol-aware AST indexing
  - Dependency graph analysis
  - Beautiful Rich UI



