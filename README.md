# 🤖 AI CLI v2.0

> **A Claude Code-level AI Terminal Assistant** - LangChain-free, Pure OpenAI Function Calling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

### 🧠 **Intelligent Code Understanding**
- **AST-Based Indexing** - Deep Python code analysis with symbol extraction
- **Semantic Search** - Find code by meaning using OpenAI embeddings  
- **Dependency Graphs** - Understand import/call relationships
- **Hybrid Retrieval** - Combines semantic, keyword, and graph-based search

### 🛡️ **Secure by Design**
- **Sandboxed Operations** - All file operations restricted to `sandbox/` directory
- **Command Validation** - Blocks dangerous terminal commands
- **Path Traversal Protection** - Prevents escaping the sandbox

### 🔧 **25 Integrated Tools**
| Category | Tools |
|----------|-------|
| **File Operations** | `create_file`, `read_file`, `write_file`, `delete_file`, `list_directory` |
| **Code Editing** | `insert_lines`, `replace_lines`, `delete_lines`, `find_and_replace` |
| **Search** | `search_code`, `search_files`, `find_symbol`, `get_file_context` |
| **Terminal** | `execute_command`, `run_python_code`, `run_python_script`, `get_system_info` |
| **Workspace** | `index_workspace`, `get_workspace_overview`, `get_related_files` |

### ⚡ **Smart Incremental Indexing**
- Auto-indexes files when created/modified/deleted
- Only re-indexes changed files on startup
- Use `/index-status` to check what needs indexing

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd Ai_in_Cli

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the `ai_cli/` directory:
```bash
echo "OPENAI_API_KEY=your-key-here" > ai_cli/.env
```

### 3. Run
```bash
# From project root
python run.py

# Or directly
cd ai_cli && python main.py
```

---

## 💬 Usage Examples

### Interactive Mode
```
You: Create a Python file that calculates fibonacci numbers

🔧 Calling create_file...
✓ Created sandbox/fibonacci.py

You: Find all functions in my code

🔧 Calling search_code...
Found 3 results for "function definitions"...

You: What terminal are we in?

🔧 Calling get_system_info...
Linux terminal, Bash shell, Python 3.11
```

### Special Commands
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/index` | Full workspace indexing |
| `/index-status` | Check index status & pending changes |
| `/stats` | Show agent statistics |
| `/tools` | List all 25 tools |
| `/workspace` | Show workspace info |
| `/clear` | Clear conversation history |
| `/exit` | Exit the assistant |

### CLI Commands
```bash
python run.py                        # Interactive mode (default)
python run.py chat "hello world"     # Single message
python run.py index                  # Index workspace
python run.py search "function"      # Search codebase
python run.py tools                  # List tools
```

---

## 📁 Project Structure

```
Ai_in_Cli/
├── run.py                    # Entry point
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── CLAUDE.md                 # AI guidelines
│
├── ai_cli/                   # Main application
│   ├── main.py               # CLI entry point with Rich UI
│   ├── config.py             # Configuration (Pydantic Settings)
│   ├── requirements.txt      # Detailed dependencies
│   │
│   ├── core/                 # Core AI functionality
│   │   ├── agent.py          # OpenAI function calling agent
│   │   ├── conversation.py   # Conversation memory
│   │   └── function_registry.py  # Tool registration
│   │
│   ├── indexing/             # Code indexing & search
│   │   ├── ast_indexer.py    # Python AST analysis
│   │   ├── semantic_indexer.py   # OpenAI embeddings
│   │   ├── dependency_graph.py   # Import/call graphs
│   │   ├── hybrid_retriever.py   # RRF fusion search
│   │   └── index_manager.py      # Incremental indexing
│   │
│   ├── storage/              # Persistence
│   │   ├── chroma_store.py   # Vector database
│   │   └── graph_store.py    # Graph persistence
│   │
│   ├── tools/                # AI Tools (25 total)
│   │   ├── file_tools.py     # CRUD file operations
│   │   ├── folder_tools.py   # Directory operations
│   │   ├── code_tools.py     # Smart editing
│   │   ├── search_tools.py   # Hybrid search
│   │   └── terminal_tools.py # Command execution
│   │
│   ├── utils/                # Utilities
│   │   ├── security.py       # Path & command validation
│   │   └── parsers.py        # AST parsing
│   │
│   └── sandbox/              # 🔒 Sandboxed workspace
│
└── docs/                     # Documentation
    └── *.md
```

---

## 🔒 Security

### Sandboxing
All file operations are restricted to the `ai_cli/sandbox/` directory:
- ✅ `sandbox/myfile.py` - Allowed
- ❌ `../config.py` - Blocked (path traversal)
- ❌ `/etc/passwd` - Blocked (absolute path outside sandbox)

### Blocked Commands
The following are blocked for safety:
- `rm -rf /`, `sudo`, `chmod 777`
- `curl | bash`, `wget | sh`
- `shutdown`, `reboot`, `halt`
- `mkfs`, `dd`, `fdisk`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       AI CLI v2.0                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Rich UI   │  │   OpenAI    │  │  Function Registry  │  │
│  │   main.py   │◄─┤   Agent     │◄─┤    25 Tools         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              Hybrid Retrieval (RRF)                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │  │
│  │  │ Semantic │  │ Keyword  │  │ Graph (NetworkX)     │ │  │
│  │  │ ChromaDB │  │ BM25-ish │  │ Dependencies         │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │                   Index Manager                        │  │
│  │  • Incremental indexing on file changes               │  │
│  │  • Startup change detection                           │  │
│  │  • File modification tracking                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Development

### Running Tests
```bash
cd ai_cli
source ../.venv/bin/activate

# Test individual components
python -c "from tools.file_tools import FileTools; print(FileTools())"
python -c "from core.agent import AIAgent; print('Agent OK')"
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model to use |
| `SANDBOX_DIR` | `sandbox` | Sandbox directory |
| `MAX_FILES_TO_SCAN` | `1000` | Limit for file scanning |

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Credits

Built with:
- [OpenAI](https://openai.com/) - GPT-4o & Embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector storage
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal UI
- [NetworkX](https://networkx.org/) - Graph analysis

---

<p align="center">
  <b>🚀 AI-powered coding, right in your terminal</b>
</p>