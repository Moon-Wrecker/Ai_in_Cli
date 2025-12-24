"""
Configuration for AI CLI Assistant
Central configuration management using Pydantic Settings
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")
    max_tokens: int = Field(default=16380, alias="MAX_TOKENS")
    
    # Agent Configuration
    max_iterations: int = Field(default=25, alias="MAX_ITERATIONS")
    max_tool_retries: int = Field(default=5, alias="MAX_TOOL_RETRIES")
    
    # Conversation Memory
    memory_window_size: int = Field(default=20, alias="MEMORY_WINDOW_SIZE")
    
    # Paths
    sandbox_dir: str = Field(default="sandbox", alias="SANDBOX_DIR")
    cache_dir: str = Field(default=".cache", alias="CACHE_DIR")
    chroma_persist_dir: str = Field(default=".chroma_db", alias="CHROMA_PERSIST_DIR")
    graph_persist_path: str = Field(default=".cache/dependency_graph.json", alias="GRAPH_PERSIST_PATH")
    
    # Indexing Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    
    # Hybrid Retrieval Weights (should sum to 1.0)
    semantic_weight: float = Field(default=0.4, alias="SEMANTIC_WEIGHT")
    keyword_weight: float = Field(default=0.3, alias="KEYWORD_WEIGHT")
    graph_weight: float = Field(default=0.3, alias="GRAPH_WEIGHT")
    
    # Security Configuration
    blocked_commands: List[str] = Field(
        default=[
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){:|:&};:",
            "chmod -R 777 /", "chown -R", "> /dev/sda", "wget", "curl -o",
            "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
            "kill -9 -1", "killall", "pkill -9", ":(){ :|:& };:",
            "mv / /dev/null", "cat /dev/zero", "fork bomb",
        ],
        alias="BLOCKED_COMMANDS"
    )
    
    allowed_extensions: List[str] = Field(
        default=[
            ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
            ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".sh", ".bash",
            ".sql", ".xml", ".csv", ".env.example", ".gitignore", ".dockerignore",
            ".dockerfile", ".makefile", ".cfg", ".ini", ".conf", ".log",
        ],
        alias="ALLOWED_EXTENSIONS"
    )
    
    ignore_patterns: List[str] = Field(
        default=[
            "__pycache__", ".git", ".vscode", ".idea", "node_modules",
            ".venv", "venv", "env", ".env", "*.pyc", "*.pyo", "*.pyd", ".DS_Store",
            "*.egg-info", "dist", "build", ".pytest_cache", ".mypy_cache",
            ".cache", ".chroma_db", ".chroma", "chroma_data", "*.log",
            "site-packages", ".tox", ".nox", "htmlcov", ".coverage",
        ],
        alias="IGNORE_PATTERNS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_sandbox_path() -> Path:
    """Get absolute path to sandbox directory"""
    settings = get_settings()
    base_dir = Path(__file__).parent
    sandbox_path = base_dir / settings.sandbox_dir
    sandbox_path.mkdir(parents=True, exist_ok=True)
    return sandbox_path.resolve()


def get_cache_path() -> Path:
    """Get absolute path to cache directory"""
    settings = get_settings()
    base_dir = Path(__file__).parent
    cache_path = base_dir / settings.cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path.resolve()


def get_chroma_path() -> Path:
    """Get absolute path to ChromaDB persistence directory"""
    settings = get_settings()
    base_dir = Path(__file__).parent
    chroma_path = base_dir / settings.chroma_persist_dir
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path.resolve()


# Application metadata
APP_NAME = "AI CLI Assistant"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Claude Code-level AI Terminal Assistant"



