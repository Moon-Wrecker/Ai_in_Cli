"""
Conversation Memory Management
Handles conversation history and context
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_cache_path


@dataclass
class Message:
    """Represents a conversation message"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None  # For assistant messages with tool calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI message format"""
        msg = {
            "role": self.role,
            "content": self.content,
        }
        
        if self.name:
            msg["name"] = self.name
        
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        
        return msg
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API message format"""
        return self.to_dict()


class ConversationMemory:
    """
    Manages conversation history with windowing and persistence.
    """
    
    def __init__(
        self,
        window_size: Optional[int] = None,
        persist: bool = True,
        session_id: Optional[str] = None,
    ):
        settings = get_settings()
        self.window_size = window_size or settings.memory_window_size
        self.persist = persist
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.messages: List[Message] = []
        self.system_message: Optional[Message] = None
        
        # Persistence path
        if persist:
            cache_path = get_cache_path()
            self.persist_path = cache_path / "conversations" / f"{self.session_id}.json"
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
    
    def set_system_message(self, content: str):
        """Set the system message"""
        self.system_message = Message(
            role="system",
            content=content,
        )
    
    def add_user_message(self, content: str):
        """Add a user message"""
        self.messages.append(Message(
            role="user",
            content=content,
        ))
        self._trim_window()
        self._save()
    
    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
    ):
        """Add an assistant message"""
        msg = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )
        self.messages.append(msg)
        self._trim_window()
        self._save()
    
    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ):
        """Add a tool result message"""
        self.messages.append(Message(
            role="tool",
            content=result,
            name=tool_name,
            tool_call_id=tool_call_id,
        ))
        self._save()
    
    def get_messages(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Get messages in OpenAI format.
        
        Args:
            include_system: Include system message
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        if include_system and self.system_message:
            messages.append(self.system_message.to_openai_format())
        
        for msg in self.messages:
            messages.append(msg.to_openai_format())
        
        return messages
    
    def get_context_summary(self, max_chars: int = 2000) -> str:
        """
        Get a summary of recent conversation context.
        Useful for providing context without full history.
        """
        summary_parts = []
        total_chars = 0
        
        # Start from recent messages
        for msg in reversed(self.messages):
            if msg.role == "user":
                part = f"User: {msg.content[:500]}"
            elif msg.role == "assistant":
                part = f"Assistant: {msg.content[:500]}"
            elif msg.role == "tool":
                part = f"Tool ({msg.name}): {msg.content[:200]}"
            else:
                continue
            
            if total_chars + len(part) > max_chars:
                break
            
            summary_parts.insert(0, part)
            total_chars += len(part)
        
        return "\n".join(summary_parts)
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self._save()
    
    def _trim_window(self):
        """Trim messages to window size"""
        if len(self.messages) > self.window_size * 2:
            # Keep tool results with their corresponding assistant messages
            # This is a simplified trimming that keeps recent messages
            self.messages = self.messages[-self.window_size * 2:]
    
    def _save(self):
        """Save conversation to disk"""
        if not self.persist:
            return
        
        try:
            data = {
                "session_id": self.session_id,
                "system_message": self.system_message.to_dict() if self.system_message else None,
                "messages": [m.to_dict() for m in self.messages],
                "saved_at": time.time(),
            }
            
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            pass  # Silently fail persistence
    
    def load(self, session_id: str) -> bool:
        """Load a previous conversation session"""
        cache_path = get_cache_path()
        load_path = cache_path / "conversations" / f"{session_id}.json"
        
        if not load_path.exists():
            return False
        
        try:
            with open(load_path, "r") as f:
                data = json.load(f)
            
            self.session_id = data["session_id"]
            
            if data.get("system_message"):
                self.system_message = Message(**data["system_message"])
            
            self.messages = [Message(**m) for m in data.get("messages", [])]
            
            return True
            
        except Exception as e:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        user_messages = sum(1 for m in self.messages if m.role == "user")
        assistant_messages = sum(1 for m in self.messages if m.role == "assistant")
        tool_messages = sum(1 for m in self.messages if m.role == "tool")
        
        total_chars = sum(len(m.content) for m in self.messages)
        
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "tool_calls": tool_messages,
            "total_characters": total_chars,
            "window_size": self.window_size,
        }


class ContextManager:
    """
    Manages additional context for the conversation.
    Provides workspace awareness to the AI.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        from config import get_sandbox_path
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self._workspace_context: Optional[str] = None
        self._last_update: float = 0
        self._update_interval: float = 60  # seconds
    
    def get_workspace_context(self, force_update: bool = False) -> str:
        """Get current workspace context"""
        current_time = time.time()
        
        if (
            force_update
            or self._workspace_context is None
            or current_time - self._last_update > self._update_interval
        ):
            self._workspace_context = self._build_workspace_context()
            self._last_update = current_time
        
        return self._workspace_context
    
    def _build_workspace_context(self) -> str:
        """Build workspace context string"""
        parts = []
        
        # List top-level structure
        parts.append(f"Workspace: {self.sandbox_path}")
        parts.append("\nTop-level contents:")
        
        try:
            items = list(self.sandbox_path.iterdir())
            folders = [i.name for i in items if i.is_dir() and not i.name.startswith(".")]
            files = [i.name for i in items if i.is_file() and not i.name.startswith(".")]
            
            for folder in sorted(folders)[:10]:
                parts.append(f"  📁 {folder}/")
            
            for file in sorted(files)[:10]:
                parts.append(f"  📄 {file}")
            
            if len(folders) > 10 or len(files) > 10:
                parts.append("  ...")
                
        except Exception:
            parts.append("  (Unable to list directory)")
        
        return "\n".join(parts)
    
    def get_file_context(self, file_path: str) -> Optional[str]:
        """Get context for a specific file"""
        try:
            resolved = self.sandbox_path / file_path
            if not resolved.exists():
                return None
            
            if resolved.suffix == ".py":
                from utils.parsers import get_file_structure
                structure = get_file_structure(resolved)
                
                parts = [f"File: {file_path}"]
                if structure.get("module_docstring"):
                    parts.append(f"Description: {structure['module_docstring'][:200]}")
                
                classes = structure.get("classes", [])
                if classes:
                    parts.append(f"Classes: {', '.join(classes)}")
                
                functions = structure.get("functions", [])
                if functions:
                    parts.append(f"Functions: {', '.join(functions)}")
                
                return "\n".join(parts)
            
            return f"File: {file_path}"
            
        except Exception:
            return None


# Singleton instances
_memory: Optional[ConversationMemory] = None
_context_manager: Optional[ContextManager] = None


def get_memory() -> ConversationMemory:
    """Get or create conversation memory"""
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


def get_context_manager() -> ContextManager:
    """Get or create context manager"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager



