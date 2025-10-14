# 🤖 AI in CLI - Quick Start Guide

## 🚀 Launch the Application

```bash
python app.py
```

## 💬 Example Commands

### File Operations
- `list all files`
- `create a file called notes.txt with hello world`
- `read the contents of README.md`
- `delete test.txt`

### Code Operations
- `write a snake game and save it`
- `create a Python hello world program`
- `show me what's in app.py`

### Search & Analysis
- `search for TODO in Python files`
- `what files do I have?`
- `analyze the structure of my code`

## 🎯 Tips

- Commands are natural language - just type what you want!
- Type `quit` or `exit` to leave
- All operations are sandboxed to the `workspace/` directory
- The AI remembers context from previous commands

## 🔧 Troubleshooting

### API Key Error
If you see "GOOGLE_API_KEY not found":
1. Create `.env` file in project root
2. Add: `GOOGLE_API_KEY=your_key_here`

### Model Not Found
The app now uses `gemini-1.5-flash-latest` - update if needed in `config.py`

## ✨ Features

- ✅ Natural language commands
- ✅ File & folder operations
- ✅ Code generation
- ✅ RAG-powered workspace awareness
- ✅ Smart file editing
- ✅ Terminal command execution (sandboxed)
- ✅ Conversation memory

Enjoy! 🎉
