# AI-Powered File System & Terminal Assistant

A terminal application that uses LangChain with Google's Gemini API to perform file operations and execute terminal commands through natural language.

## Features

🤖 **Natural Language Interface**: Control your file system and terminal using plain English commands  
📁 **Complete File Operations**: Create, read, write, edit, delete files and folders  
💻 **Safe Terminal Execution**: Execute terminal commands with built-in safety restrictions  
🔍 **Smart Search**: Find text across multiple files with context  
📊 **Code Analysis**: Analyze code structure and dependencies  
🌐 **Cross-Platform**: Works on Windows (CMD/PowerShell), Linux, and macOS  
💾 **Safe Operations**: Built-in safety checks and command validation  
🧠 **Conversation Memory**: Maintains context across commands  
🔒 **Workspace Sandboxing**: All file operations are isolated to a designated workspace directory  

## Security & Safety Features

✅ **Safe Command Execution**: Only allows safe, non-destructive terminal commands  
🛡️ **Security Features**: Blocks dangerous operations like system shutdown, formatting, etc.  
� **Workspace Isolation**: All file operations are restricted to the `./workspace/` directory  
�🔧 **Cross-Platform Support**: Automatically detects and works with your OS and shell  
💡 **Smart Suggestions**: Suggests appropriate commands based on your intent  
📋 **System Information**: Get comprehensive system details and environment info  

## Workspace Sandboxing

The AI assistant operates in a sandboxed environment to protect your system and application source code:

- **Isolated Operations**: All file and directory operations are restricted to the `./workspace/` directory
- **Source Code Protection**: The assistant cannot modify or interfere with its own source code files
- **Safe Exploration**: You can experiment freely without affecting your system
- **Automatic Setup**: The workspace directory is created automatically on first run
- **Clear Boundaries**: The assistant will inform you if you try to access files outside the workspace  

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Key

1. Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### 3. Run the Assistant

#### Simple Version (Recommended for testing):
```bash
python simple_assistant.py
```

#### Full Version with Rich Interface:
```bash
python ai_assistant.py interactive
```

#### Single Command Execution:
```bash
python ai_assistant.py execute "list all files in the current directory"
```

### 4. Workspace Configuration (Optional)

By default, the assistant creates a `./workspace/` directory for your files. You can customize this:

1. Add to your `.env` file:
   ```
   WORKSPACE_DIR=my_custom_workspace
   ```
2. Or use an absolute path:
   ```
   WORKSPACE_DIR=/home/user/my_ai_workspace
   ```

## Usage Examples

### Basic File Operations
```
💬 You: Create a new file called notes.txt with some sample content
💬 You: Read the contents of README.md
💬 You: List all files in the tools directory
💬 You: Create a folder called project_docs
```

### Advanced Operations
```
💬 You: Search for the word "TODO" in all Python files
💬 You: Analyze the code structure of app.py
💬 You: Find all files modified today
💬 You: Create a backup of important.txt
```

### Content Creation
```
💬 You: Create a Python script that prints hello world
💬 You: Write a simple HTML page in index.html
💬 You: Add a new function to calculate fibonacci numbers in utils.py
```

### Terminal Operations (New!)
```
💬 You: What's my current directory?
💬 You: Show system information
💬 You: List all running processes
💬 You: Check Python version
💬 You: Get network configuration
💬 You: Suggest commands for listing files
```

## Available Commands

The assistant can understand natural language for these operations:

| Operation | Examples |
|-----------|----------|
| **List Directory** | "show me all files", "list contents of docs folder" |
| **Read File** | "read config.py", "show me the contents of notes.txt" |
| **Create File** | "create a new file called test.py", "make a README file" |
| **Write File** | "write 'hello world' to greeting.txt" |
| **Append File** | "add a new line to log.txt" |
| **Delete File** | "delete old_file.txt", "remove temp files" |
| **Create Directory** | "create a folder called images", "make new directory" |
| **Search Files** | "find 'TODO' in all files", "search for function names" |
| **Terminal Commands** | "what's my current directory?", "list running processes" |
| **System Info** | "show system information", "what OS am I running?" |
| **Command Help** | "suggest commands for listing files", "how to check disk space" |
| **Analyze Code** | "analyze the structure of main.py" |

## Project Structure

```
ai-in-cli/
├── ai_assistant.py      # Full-featured terminal app with rich UI
├── simple_assistant.py  # Minimal version for quick testing
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── tools/             # File system utilities
│   └── file_fol_tools.py
└── README.md          # This file
```

## Architecture

### Components

1. **FileSystemManager** (`tools/file_fol_tools.py`)
   - Core file system operations
   - Safety checks and path validation
   - Error handling and logging

2. **AI Assistant** (`ai_assistant.py` / `simple_assistant.py`)
   - LangChain integration with Gemini API
   - Natural language processing
   - Command interpretation and execution

3. **LangChain Tools**
   - Wraps FileSystemManager methods as LangChain tools
   - Handles input/output formatting
   - Provides structured interface for AI agent

### How It Works

1. **User Input**: Natural language command via terminal
2. **AI Processing**: Gemini analyzes intent and determines required operations
3. **Tool Selection**: LangChain agent selects appropriate file system tools
4. **Execution**: FileSystemManager performs the actual file operations
5. **Response**: Results formatted and returned to user

## Troubleshooting

### Common Issues

**API Key Error**:
```
 GOOGLE_API_KEY not found in environment variables
```
- Solution: Make sure you've created `.env` file with your API key

**Import Error**:
```
ModuleNotFoundError: No module named 'langchain'
```
- Solution: Install requirements with `pip install -r requirements.txt`

**Permission Error**:
```
PermissionError: [Errno 13] Permission denied
```
- Solution: Check file/folder permissions, run with appropriate privileges

## Advanced Features

### Memory and Context
The assistant maintains conversation memory, so you can refer to previous operations:
```
💬 You: Create a file called data.json
💬 You: Now add some sample data to that file
💬 You: Read it back to verify the content
```

### Safety Features
- Path validation and sandboxing
- Backup creation for destructive operations  
- Confirmation prompts for dangerous operations
- Error recovery and rollback capabilities

### Future Enhancements
- [ ] RAG (Retrieval Augmented Generation) for project context
- [ ] Git integration for version control
- [ ] Plugin system for custom operations
- [ ] Web interface for remote access
- [ ] Batch operation support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the example commands for usage patterns