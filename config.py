"""
Configuration settings for AI File Assistant
"""

import os
from pathlib import Path

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration - Using OpenAI
LLM_MODEL = "gpt-4o-2024-08-06"  # Optimized for tool/function calling
LLM_TEMPERATURE = 0.1  # Lower = more deterministic, higher = more creative
MAX_ITERATIONS = 20  # Allow complex multi-step operations
MAX_EXECUTION_TIME = 90  # Generous timeout for complex tasks

# Memory Configuration
MEMORY_WINDOW_SIZE = 25  # Larger context window for better understanding

# File System Configuration
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "workspace")
BACKUP_DIR = os.getenv("BACKUP_DIR", ".code_backups")
DEFAULT_IGNORE_PATTERNS = [
    "__pycache__",
    ".git", 
    ".vscode",
    ".idea",
    "node_modules",
    "venv",
    "env",
    ".env",
    "*.pyc",
    "*.pyo", 
    "*.pyd"
]

def get_workspace_path():
    """Get the absolute path to the workspace directory"""
    workspace_path = Path(WORKSPACE_DIR)
    
    # Create workspace directory if it doesn't exist
    workspace_path.mkdir(exist_ok=True)
    
    return workspace_path.resolve()

# Application Configuration
APP_NAME = "AI File Assistant"
APP_VERSION = "2.0.0"

# RAG Configuration - Outside workspace to keep it clean
RAG_CACHE_DIR = Path(".rag_cache")  # Project root, not workspace
RAG_DB_DIR = Path(".rag_db")  # Project root, not workspace
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Protected Files - App should NEVER modify these
PROTECTED_PATTERNS = [
    ".*",  # All hidden files/folders (starting with .)
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".git*",
    ".env*",
]
