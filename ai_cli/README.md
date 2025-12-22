# 🤖 AI CLI v2.0

**Claude Code-level AI Terminal Assistant** - Built without LangChain using direct OpenAI function calling.

---

## ✨ Features

- 🔧 **25 Built-in Tools** - File ops, code editing, terminal commands, search
- 🔍 **Hybrid Search** - Semantic + Keyword + Graph with RRF fusion
- 🌳 **AST Indexing** - Deep Python code understanding
- 📊 **Dependency Graph** - Import and call relationship tracking
- 🔒 **Sandboxed** - All operations restricted to sandbox directory
- 💾 **ChromaDB** - Persistent vector storage for semantic search
- 🎨 **Rich UI** - Beautiful terminal interface

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 2. Installation

```bash
cd ai_cli

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the `ai_cli` directory:

```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### 4. Run

```bash
python main.py
```

---

## 📖 Usage Guide

### Interactive Mode (Recommended)

```bash
python main.py
```

You'll see:
```
╔════════════════════════════════════════════════════╗
║               🤖 AI CLI Assistant                  ║
║          Claude Code-level AI Assistant            ║
║                    v2.0.0                          ║
╚════════════════════════════════════════════════════╝

📁 Workspace: /path/to/ai_cli/sandbox

Quick Commands:
  • Type your request in natural language
  • /help     - Show all commands
  • /clear    - Clear conversation
  • /index    - Index workspace for search
  • /exit     - Exit the assistant

You: 
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/clear` | Clear conversation history |
| `/index` | Full workspace indexing |
| `/index-status` | Check index status & pending changes |
| `/stats` | Show agent statistics (tokens, calls) |
| `/tools` | List all 25 available tools |
| `/workspace` | Show sandbox contents |
| `/exit` | Exit the assistant |

### CLI Commands

```bash
# Interactive mode (default)
python main.py

# Single message
python main.py chat "Create a hello.py file"

# Index workspace
python main.py index
python main.py index --force    # Force re-index

# Search codebase
python main.py search "function"
python main.py search "auth" -n 20

# List tools
python main.py tools

# Show version
python main.py version
```

---

## 💬 Example Requests

### File Operations
```
Create a file called utils.py with a helper function
Read the contents of config.py
Delete the old_file.txt
Show me lines 10-50 of main.py
```

### Code Editing
```
Add a new function at line 25 of app.py
Replace lines 10-15 with updated code
Find and replace "old_name" with "new_name" in utils.py
Analyze the structure of module.py
```

### Directory Operations
```
List all files in the workspace
Create a folder called tests
Find all Python files
Show the directory tree
```

### Search & Analysis
```
Search for "database connection" in the codebase
Find all functions that handle authentication
Show me code related to user validation
```

### Terminal Commands
```
Run ls -la in the workspace
Execute the test script test.py
Show system information
```

---

## 🛠️ Available Tools (25)

### File Tools
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `read_file_lines` | Read specific lines with line numbers |
| `create_file` | Create new file |
| `write_file` | Overwrite file |
| `delete_file` | Delete file |
| `get_file_info` | Get file metadata |

### Folder Tools
| Tool | Description |
|------|-------------|
| `list_directory` | List directory contents |
| `create_directory` | Create directory |
| `delete_directory` | Delete directory |
| `find_files` | Find files by pattern |
| `get_directory_info` | Get directory stats |

### Code Tools
| Tool | Description |
|------|-------------|
| `insert_lines` | Insert at line number |
| `replace_lines` | Replace line range |
| `delete_lines` | Delete line range |
| `find_and_replace` | Find/replace with regex |
| `apply_edit` | Precise content edit |
| `analyze_code` | Analyze Python structure |

### Terminal Tools
| Tool | Description |
|------|-------------|
| `execute_command` | Run shell command |
| `run_python_code` | Execute Python code |
| `run_python_script` | Run Python file |
| `get_system_info` | Get system info |

### Search Tools
| Tool | Description |
|------|-------------|
| `search_codebase` | Hybrid search |
| `search_symbol` | Find by symbol name |
| `index_workspace` | Index for search |
| `get_workspace_overview` | Workspace stats |

---

## 📁 Project Structure

```
ai_cli/
├── main.py                  # Entry point
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── .env                     # API key (create this)
│
├── core/                    # Core components
│   ├── agent.py             # OpenAI function calling
│   ├── conversation.py      # Memory management
│   └── function_registry.py # Tool definitions
│
├── indexing/                # Code indexing
│   ├── ast_indexer.py       # Python AST parsing
│   ├── semantic_indexer.py  # OpenAI embeddings
│   ├── dependency_graph.py  # Import/call graphs
│   └── hybrid_retriever.py  # RRF fusion search
│
├── tools/                   # Tool implementations
│   ├── file_tools.py        # File operations
│   ├── folder_tools.py      # Directory operations
│   ├── code_tools.py        # Code editing
│   ├── terminal_tools.py    # Command execution
│   └── search_tools.py      # Search interface
│
├── storage/                 # Persistence
│   ├── chroma_store.py      # Vector database
│   └── graph_store.py       # Dependency graph
│
├── utils/                   # Utilities
│   ├── security.py          # Sandboxing
│   └── parsers.py           # Python parser
│
└── sandbox/                 # ⚠️ ALL operations happen here
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional - Model settings
OPENAI_MODEL=gpt-4o           # Default model
TEMPERATURE=0.1               # Response creativity
MAX_TOKENS=28000              # Max response tokens

# Optional - Agent settings  
MAX_ITERATIONS=25             # Max tool call iterations
MEMORY_WINDOW_SIZE=20         # Conversation memory size

# Optional - Search weights (must sum to 1.0)
SEMANTIC_WEIGHT=0.4
KEYWORD_WEIGHT=0.3
GRAPH_WEIGHT=0.3
```

---

## 🔒 Security

### Sandboxing
- **All file operations** are restricted to the `sandbox/` directory
- Path traversal attacks (e.g., `../`) are blocked
- Absolute paths outside sandbox are rejected

### Command Safety
- Dangerous commands are blocked: `rm -rf /`, `mkfs`, `dd if=`, etc.
- Commands are validated before execution
- Restricted environment for command execution

---

## 🔍 Search System

### Hybrid Retrieval
The search system combines three methods:

1. **Semantic Search** - Uses OpenAI embeddings + ChromaDB
2. **Keyword Search** - BM25-like term matching
3. **Graph Search** - Symbol relationship traversal

Results are combined using **Reciprocal Rank Fusion (RRF)**:
```
score = Σ (weight / (k + rank))
```

### Indexing
Before using search, index the workspace:
```
/index
```
Or from CLI:
```bash
python main.py index
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
Create `.env` file:
```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### "File not found" errors
All paths are relative to `sandbox/`:
```
# Correct:
read_file test.py

# Wrong:
read_file /home/user/test.py
```

### Search returns no results
Index the workspace first:
```
/index
```

### Command blocked
Some commands are blocked for safety. The AI will explain what's blocked.

---

## 🔄 Smart Indexing

The AI CLI features **smart incremental indexing** to keep the LLM aware of file changes.

### How It Works

| Trigger | What Happens |
|---------|--------------|
| **Startup** | Checks for new/modified/deleted files, indexes only changes |
| **File Create** | Auto-indexes new Python files immediately |
| **File Modify** | Auto-re-indexes modified Python files |
| **File Delete** | Removes deleted files from index |
| **`/index`** | Full re-index of entire workspace |

### Check Index Status

```
/index-status
```

Shows:
- Number of indexed files
- Last full index time
- Pending new/modified/deleted files
- Whether re-indexing is needed

### How Files Are Tracked

The system tracks:
- **Modification time** (`mtime`) - detects edits
- **File size** - quick change detection
- **Index timestamp** - knows what's stale

State is persisted in `.cache/index_state.json`.

### What Gets Indexed

- ✅ Python files (`.py`) - Full AST + semantic indexing
- ✅ Other code files - Semantic indexing only
- ❌ Binary files - Skipped
- ❌ `.venv`, `node_modules`, etc. - Always ignored

---

## ⚡ Performance Notes

### Lazy Loading
Heavy components (ChromaDB, embeddings, AST parsers) are loaded **on-demand**, not at startup:
- SearchTools initializes instantly
- HybridRetriever loads only when search is performed
- Vector stores connect only when needed

### File Limits
To prevent resource exhaustion:
- File scans limited to **500 files** for search
- Workspace overview scans **max 1000 files**
- Directories like `.venv`, `node_modules`, `__pycache__` are always skipped

### Ignored Directories
These directories are automatically skipped during indexing/search:
```
.venv, venv, node_modules, __pycache__, .git, .cache, 
.chroma_db, .mypy_cache, .pytest_cache, site-packages
```

### Resource-Intensive Operations
These operations may take time on large codebases:
- `/index` - Full workspace indexing (runs AST + semantic + graph)
- First semantic search (loads embeddings model)

Tip: Keep your sandbox directory focused on your project files only.

---

## 📊 Statistics

Track usage with `/stats`:
```
╭─────────────── Agent Statistics ───────────────╮
│ Metric                │ Value                  │
├───────────────────────┼────────────────────────┤
│ Model                 │ gpt-4o                 │
│ Temperature           │ 0.1                    │
│ Total Tokens Used     │ 15234                  │
│ Tool Calls Made       │ 47                     │
│ Available Tools       │ 25                     │
│ Messages in Memory    │ 12                     │
╰───────────────────────┴────────────────────────╯
```

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🙏 Credits

- OpenAI GPT-4 for AI capabilities
- ChromaDB for vector storage
- NetworkX for graph operations
- Rich for beautiful terminal UI

---

**Made with ❤️ for developers who love the terminal**
