# AI CLI Assistant

A natural language-powered terminal assistant for file operations and workspace management using Google Gemini and LangChain.

## Overview

AI CLI Assistant provides an intelligent command-line interface that interprets natural language commands to perform file operations, code analysis, and workspace management tasks. The system leverages Google's Gemini AI model with RAG (Retrieval-Augmented Generation) for context-aware assistance.

## Key Features

- **Natural Language Processing**: Execute file operations using conversational commands
- **RAG Integration**: Semantic search and workspace understanding through vector embeddings
- **Smart Code Editing**: Advanced file modification with line-level precision
- **Workspace Sandboxing**: All operations isolated to designated workspace directory
- **Cross-Platform**: Compatible with Windows, Linux, and macOS
- **Conversation Memory**: Maintains context across multiple interactions  

## Installation

### Prerequisites

- Python 3.8 or higher
- Google AI API key

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai_cli_assistant.git
   cd ai_cli_assistant
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure API key:

   Create a `.env` file in the project root:

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

   Obtain your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

Run the assistant in interactive mode:

```bash
python app.py
```

The assistant provides access to file operations, code analysis, and workspace management through natural language commands.

## Command Examples

**File Operations:**

- "List all files in the current directory"
- "Create a new file called notes.txt with sample content"
- "Read the contents of README.md"
- "Delete the file test.txt"

**Code Analysis:**

- "Analyze the structure of app.py"
- "Search for 'TODO' in all Python files"
- "Show me the functions in ai_assistant.py"

**Workspace Management:**

- "Create a folder called project_docs"
- "Find all files modified today"
- "Get an overview of the workspace"

## Architecture

### Core Components

**AI Assistant** (`ai_assistant.py`)

- LangChain integration with Google Gemini
- Natural language command processing
- Conversation memory management

**RAG System** (`tools/rag_system.py`)

- Workspace indexing using ChromaDB
- Semantic code search
- Context-aware assistance

**Smart Editor** (`tools/smart_editor.py`)

- Advanced file modification capabilities
- Line-level precision editing
- Code structure analysis

**File System Tools** (`tools/file_fol_tools.py`)

- Core file operations
- Path validation and security
- Workspace sandboxing

### Workflow

1. User inputs natural language command
2. Gemini AI interprets intent and selects appropriate tools
3. LangChain agent executes file operations through tool wrappers
4. Results are formatted and returned to user
5. Conversation context is maintained for follow-up commands

## Security

All file operations are restricted to the designated workspace directory (default: `./workspace/`). The system includes:

- Path validation and sandboxing
- Protection against directory traversal attacks
- Safe command execution with restricted scope
- Automatic workspace creation and isolation

## Technical Stack

- **AI Model**: Google Gemini 1.5 Flash
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **UI**: Rich (Python library)
- **Language**: Python 3.8+

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## License

MIT License - see [LICENSE](LICENSE) file for details.
