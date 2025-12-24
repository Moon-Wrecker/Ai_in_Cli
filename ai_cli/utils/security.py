"""
Security utilities for sandboxing and path validation
Ensures all file operations are restricted to the sandbox directory
"""

import os
import re
from pathlib import Path
from typing import Tuple, Optional, List
from functools import wraps

from config import get_settings, get_sandbox_path


class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass


class PathValidator:
    """Validates and sanitizes file paths for sandbox operations"""
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.settings = get_settings()
    
    def resolve_path(self, user_path: str) -> Path:
        """
        Resolve a user-provided path to an absolute path within the sandbox.
        
        Args:
            user_path: Path string from user (relative or absolute)
            
        Returns:
            Resolved absolute Path within sandbox
            
        Raises:
            SecurityError: If path attempts to escape sandbox
        """
        # Handle empty or None paths
        if not user_path or user_path.strip() == "":
            return self.sandbox_path
        
        user_path = user_path.strip()
        
        # Remove any leading slashes to treat as relative
        if user_path.startswith("/"):
            # Check if it's already inside sandbox
            abs_path = Path(user_path).resolve()
            if self._is_within_sandbox(abs_path):
                return abs_path
            # Otherwise, strip leading slash and treat as relative
            user_path = user_path.lstrip("/")
        
        # Resolve relative to sandbox
        resolved = (self.sandbox_path / user_path).resolve()
        
        # Verify it's within sandbox
        if not self._is_within_sandbox(resolved):
            raise SecurityError(
                f"Path '{user_path}' attempts to escape sandbox. "
                f"All operations must be within: {self.sandbox_path}"
            )
        
        return resolved
    
    def _is_within_sandbox(self, path: Path) -> bool:
        """Check if a path is within the sandbox directory"""
        try:
            path.relative_to(self.sandbox_path)
            return True
        except ValueError:
            return False
    
    def validate_filename(self, filename: str) -> Tuple[bool, str]:
        """
        Validate a filename for safety.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check for path traversal attempts
        if ".." in filename or filename.startswith("/"):
            return False, "Filename contains invalid characters"
        
        # Check for null bytes
        if "\x00" in filename:
            return False, "Filename contains null bytes"
        
        # Check for special characters that could be dangerous
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
        for char in dangerous_chars:
            if char in filename:
                return False, f"Filename contains invalid character: {char}"
        
        return True, "Valid filename"
    
    def get_relative_path(self, absolute_path: Path) -> str:
        """Get path relative to sandbox for display"""
        try:
            return str(absolute_path.relative_to(self.sandbox_path))
        except ValueError:
            return str(absolute_path)


class CommandValidator:
    """Validates terminal commands for safety"""
    
    def __init__(self):
        self.settings = get_settings()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for command validation"""
        self.blocked_patterns = []
        for cmd in self.settings.blocked_commands:
            # Escape special regex chars and create pattern
            pattern = re.escape(cmd).replace(r"\ ", r"\s+")
            self.blocked_patterns.append(re.compile(pattern, re.IGNORECASE))
    
    def validate_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Validate a command for safety.
        
        Returns:
            Tuple of (is_safe, risk_level, message)
            risk_level: "safe", "warning", "blocked"
        """
        if not command or not command.strip():
            return False, "blocked", "Empty command"
        
        command = command.strip()
        
        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(command):
                return False, "blocked", f"Command matches blocked pattern"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r"sudo\s+rm", "blocked", "Sudo rm commands are blocked"),
            (r">\s*/dev/", "blocked", "Writing to /dev/ is blocked"),
            (r"\|\s*sh\b", "warning", "Piping to shell may be dangerous"),
            (r"\|\s*bash\b", "warning", "Piping to bash may be dangerous"),
            (r"eval\s+", "warning", "Eval commands may be dangerous"),
            (r"exec\s+", "warning", "Exec commands may be dangerous"),
            (r"rm\s+-r", "warning", "Recursive delete - use with caution"),
            (r"chmod\s+777", "warning", "Setting permissions to 777 is insecure"),
        ]
        
        for pattern, level, msg in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                if level == "blocked":
                    return False, level, msg
                # For warnings, we allow but flag
                return True, level, msg
        
        # Safe commands whitelist (for faster approval)
        safe_prefixes = [
            "ls", "cat", "head", "tail", "grep", "find", "echo", "pwd",
            "cd", "mkdir", "touch", "cp", "mv", "wc", "sort", "uniq",
            "python", "python3", "pip", "pip3", "node", "npm", "git status",
            "git log", "git diff", "git branch", "tree", "file", "stat",
        ]
        
        for prefix in safe_prefixes:
            if command.startswith(prefix):
                return True, "safe", "Command is in safe list"
        
        # Default: allow with warning for unknown commands
        return True, "warning", "Command not in known safe list - proceeding with caution"
    
    def get_safe_environment(self) -> dict:
        """Get a restricted environment for command execution"""
        # Start with minimal environment
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "TERM": os.environ.get("TERM", "xterm-256color"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        }
        
        # Add Python-related paths
        if "VIRTUAL_ENV" in os.environ:
            safe_env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
        
        return safe_env


def require_sandbox(func):
    """Decorator to ensure function operates within sandbox"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract path argument (first positional or 'path' keyword)
        path_arg = None
        if args:
            path_arg = args[0] if isinstance(args[0], (str, Path)) else None
        if path_arg is None:
            path_arg = kwargs.get('path') or kwargs.get('file_path') or kwargs.get('filepath')
        
        if path_arg:
            validator = PathValidator()
            # This will raise SecurityError if path is invalid
            validator.resolve_path(str(path_arg))
        
        return func(*args, **kwargs)
    return wrapper


# Module-level instances for convenience
_path_validator: Optional[PathValidator] = None
_command_validator: Optional[CommandValidator] = None


def get_path_validator() -> PathValidator:
    """Get or create path validator instance"""
    global _path_validator
    if _path_validator is None:
        _path_validator = PathValidator()
    return _path_validator


def get_command_validator() -> CommandValidator:
    """Get or create command validator instance"""
    global _command_validator
    if _command_validator is None:
        _command_validator = CommandValidator()
    return _command_validator



