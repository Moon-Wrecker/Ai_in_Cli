"""
Configuration settings for AI File Assistant
"""

import os
from pathlib import Path

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Configuration
LLM_MODEL = "gemini-1.5-flash-latest"  # Updated to latest stable model
LLM_TEMPERATURE = 0.1
MAX_ITERATIONS = 15

# Memory Configuration
MEMORY_WINDOW_SIZE = 10

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
APP_VERSION = "1.0.0"

# RAG Configuration (for future use)
VECTOR_STORE_PATH = Path("./vector_store")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
