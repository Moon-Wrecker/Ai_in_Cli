<div align="center">

# 🤖 AI CLI v2.0

### A Claude Code-level AI Terminal Assistant

**LangChain-free • Pure OpenAI Function Calling • Multi-Language Support**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Features](#-features) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Architecture](#-architecture) • [Security](#-security)

</div>

---

## 🏗️ Architecture

<div align="center">

![AI CLI Architecture](docs/images/architecture.png)

*Complete system architecture showing all components and their interactions*

</div>

<details>
<summary><b>📊 Data Flow</b></summary>

```
User Request → CLI Interface → AI Agent → OpenAI API
                                  ↓
                           Function Call
                                  ↓
                    Tools (File/Code/Terminal/Search)
                                  ↓
                         Index Manager (auto-update)
                                  ↓
                    Storage (ChromaDB/Graph/JSON)
                                  ↓
                           Response → User
```

</details>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Intelligent Code Understanding
- **AST-Based Indexing** — Deep code analysis with symbol extraction
- **Multi-Language Support** — Python, JavaScript, TypeScript, HTML, CSS
- **Semantic Search** — Find code by meaning using OpenAI embeddings
- **Dependency Graphs** — Understand import/call relationships
- **Hybrid Retrieval** — RRF fusion of semantic + keyword + graph search

</td>
<td width="50%">

### 🎮 Advanced Capabilities
- **GUI App Support** — Run pygame, tkinter, PyQt applications
- **Smart Editing** — Line-based code modifications
- **Auto-Indexing** — Incremental updates on file changes
- **25 Integrated Tools** — Complete development toolkit
- **Rich Terminal UI** — Beautiful, interactive interface

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Secure by Design
- **Sandboxed Operations** — All files restricted to `sandbox/`
- **Command Validation** — Blocks dangerous terminal commands
- **Path Traversal Protection** — Prevents escaping the sandbox
- **Safe Defaults** — Security-first configuration

</td>
<td width="50%">

### ⚡ Performance Optimized
- **Lazy Loading** — Components initialize on-demand
- **File Scan Limits** — Prevents resource exhaustion
- **Smart Caching** — Avoids redundant operations
- **Incremental Indexing** — Only processes changed files

</td>
</tr>
</table>

---

## 🔧 Tools Overview

| Category | Tools | Description |
|:--------:|-------|-------------|
| 📁 **File** | `create_file` `read_file` `write_file` `delete_file` `append_file` | CRUD operations for files |
| 📂 **Folder** | `list_directory` `create_directory` `get_tree` `get_folder_structure` | Directory management |
| ✏️ **Code** | `insert_lines` `replace_lines` `delete_lines` `find_and_replace` | Smart code editing |
| 🔍 **Search** | `search_code` `search_files` `find_symbol` `get_file_context` | Hybrid codebase search |
| 💻 **Terminal** | `execute_command` `run_python_code` `run_python_script` `get_system_info` | Command execution |
| 🗂️ **Workspace** | `index_workspace` `get_workspace_overview` `get_related_files` | Project understanding |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/Moon-Wrecker/Ai_in_Cli.git
cd Ai_in_Cli

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Create .env file in ai_cli directory
echo "OPENAI_API_KEY=sk-your-key-here" > ai_cli/.env
```

### Run

```bash
# Start the assistant
python run.py
```

---

## 💬 Usage

### Interactive Mode

```
┌──────────────────────────────────────────────────────────────┐
│  🤖 AI CLI v2.0 - Your Intelligent Terminal Assistant        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  You: Create a snake game with pygame                        │
│                                                              │
│  🔧 Calling create_file...                                   │
│  ✓ Created sandbox/snake_game.py (145 lines)                 │
│                                                              │
│  🎮 GUI detected! Run with: python3 sandbox/snake_game.py    │
│                                                              │
│  You: Find all classes in my codebase                        │
│                                                              │
│  🔧 Calling search_code...                                   │
│  Found 12 classes across 8 files...                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Special Commands

| Command | Description |
|:--------|:------------|
| `/help` | Show all available commands |
| `/tools` | List all 25 tools with descriptions |
| `/index` | Force full workspace indexing |
| `/index-status` | Check index status & pending changes |
| `/workspace` | Show workspace overview |
| `/stats` | Display agent statistics |
| `/clear` | Clear conversation history |
| `/exit` | Exit the assistant |

### CLI Commands

```bash
python run.py                        # Interactive mode (default)
python run.py chat "create a todo app"  # Single message
python run.py index                  # Index workspace
python run.py search "function"      # Search codebase
python run.py tools                  # List all tools
```

---

## 📁 Project Structure

```
Ai_in_Cli/
├── 📄 run.py                    # Entry point
├── 📄 requirements.txt          # Dependencies
├── 📄 README.md                 # This file
│
├── 📂 ai_cli/                   # Main application
│   ├── 📄 main.py               # CLI with Rich UI
│   ├── 📄 config.py             # Pydantic Settings
│   │
│   ├── 📂 core/                 # AI functionality
│   │   ├── agent.py             # OpenAI function calling
│   │   ├── conversation.py      # Message history
│   │   └── function_registry.py # Tool registration
│   │
│   ├── 📂 indexing/             # Code indexing
│   │   ├── ast_indexer.py       # Multi-language AST
│   │   ├── semantic_indexer.py  # OpenAI embeddings
│   │   ├── dependency_graph.py  # Import graphs
│   │   ├── hybrid_retriever.py  # RRF fusion
│   │   └── index_manager.py     # Incremental indexing
│   │
│   ├── 📂 tools/                # 25 AI Tools
│   │   ├── file_tools.py        # File operations
│   │   ├── folder_tools.py      # Directory ops
│   │   ├── code_tools.py        # Smart editing
│   │   ├── search_tools.py      # Hybrid search
│   │   └── terminal_tools.py    # Command + GUI
│   │
│   ├── 📂 utils/                # Utilities
│   │   ├── security.py          # Path validation
│   │   └── parsers.py           # Multi-language parsers
│   │
│   └── 📂 sandbox/              # 🔒 Sandboxed workspace
│
└── 📂 docs/                     # Documentation
    └── 📂 images/
        └── architecture.png     # Architecture diagram
```

---

## 🔒 Security

<table>
<tr>
<td>

### ✅ Allowed
```
sandbox/myfile.py
sandbox/src/app.js
sandbox/styles/main.css
```

</td>
<td>

### ❌ Blocked
```
../config.py          # Path traversal
/etc/passwd           # System files
~/.ssh/id_rsa         # Sensitive data
```

</td>
</tr>
</table>

### Blocked Commands

| Category | Commands |
|----------|----------|
| **Destructive** | `rm -rf /`, `mkfs`, `dd`, `fdisk` |
| **Privilege** | `sudo`, `su`, `chmod 777` |
| **Remote Exec** | `curl \| bash`, `wget \| sh` |
| **System** | `shutdown`, `reboot`, `halt` |

---

## 🌐 Supported Languages

| Language | Writing | Features |
|:--------:|:-------:|----------|
| 🐍 Python | ✅ | Classes, functions, imports, decorators |
| 📜 JavaScript | ✅ | Functions, classes, arrow functions |
| 📘 TypeScript | ✅ | + Interfaces, types, enums |
| 🌐 HTML | ✅ | Tags, IDs, classes, components |
| 🎨 CSS/SCSS | ✅ | Selectors, variables, keyframes |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model for chat |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `SANDBOX_DIR` | `sandbox` | Restricted workspace |
| `MAX_FILES_TO_SCAN` | `1000` | File scan limit |

---

## 🧪 Development

```bash
# Activate environment
source .venv/bin/activate

# Test components
cd ai_cli
python -c "from core.agent import AIAgent; print('✓ Agent OK')"
python -c "from tools.file_tools import FileTools; print('✓ Tools OK')"
python -c "from indexing.hybrid_retriever import HybridRetriever; print('✓ Indexing OK')"
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Built With

<p align="center">
<a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/></a>
<a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge" alt="ChromaDB"/></a>
<a href="https://rich.readthedocs.io/"><img src="https://img.shields.io/badge/Rich-Terminal_UI-blue?style=for-the-badge" alt="Rich"/></a>
<a href="https://networkx.org/"><img src="https://img.shields.io/badge/NetworkX-Graphs-orange?style=for-the-badge" alt="NetworkX"/></a>
</p>

---

<div align="center">

**🚀 AI-powered coding, right in your terminal**

Made with ❤️ by [Moon-Wrecker](https://github.com/Moon-Wrecker)

</div>
