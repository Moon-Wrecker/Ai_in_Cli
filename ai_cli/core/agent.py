"""
AI Agent - OpenAI Function Calling Loop
Core agent implementation without LangChain
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path, APP_NAME, APP_VERSION
from core.conversation import ConversationMemory, ContextManager, get_memory, get_context_manager
from core.function_registry import FunctionRegistry, get_registry


@dataclass
class AgentResponse:
    """Response from the agent"""
    content: str
    tool_calls_made: int = 0
    tokens_used: int = 0
    iterations: int = 0
    duration_seconds: float = 0


class AIAgent:
    """
    AI Agent using OpenAI function calling.
    Implements a ReAct-style loop for tool execution.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        memory: Optional[ConversationMemory] = None,
        registry: Optional[FunctionRegistry] = None,
    ):
        self.settings = get_settings()
        self.model = model or self.settings.openai_model
        self.temperature = temperature if temperature is not None else self.settings.temperature
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        
        # Initialize components
        self.memory = memory or get_memory()
        self.registry = registry or get_registry()
        self.context_manager = get_context_manager()
        
        # Set system message
        self._set_system_message()
        
        # Stats
        self.total_tokens_used = 0
        self.total_tool_calls = 0
    
    def _set_system_message(self):
        """Set the system message for the agent"""
        workspace_context = self.context_manager.get_workspace_context()
        
        system_prompt = f"""You are {APP_NAME} v{APP_VERSION}, an advanced AI coding assistant.

## Capabilities
You can help users with:
- File operations: read, create, write, delete files
- Directory operations: list, create, delete directories
- Code editing: insert, replace, delete lines; find and replace
- Code analysis: analyze Python code structure
- Terminal commands: execute safe commands
- Semantic search: search codebase by meaning
- Code navigation: find symbols, related code

## Workspace
{workspace_context}

## Guidelines
1. Always read files before editing them to understand current content
2. Use specific line numbers when making edits
3. Prefer incremental edits over full file rewrites
4. Validate paths exist before operations
5. Use search to find relevant code before making changes
6. Explain what you're doing and why

## Safety
- All operations are sandboxed to the workspace directory
- Some commands may be blocked for safety
- Always confirm destructive operations

Be helpful, precise, and efficient. When you need information, use the available tools."""

        self.memory.set_system_message(system_prompt)
    
    def chat(self, user_message: str) -> AgentResponse:
        """
        Process a user message and return a response.
        Implements the main agent loop with tool calling.
        
        Args:
            user_message: The user's input
            
        Returns:
            AgentResponse with the final response and stats
        """
        start_time = time.time()
        iterations = 0
        tool_calls_made = 0
        tokens_used = 0
        
        # Add user message to memory
        self.memory.add_user_message(user_message)
        
        # Get tool schemas
        tools = self.registry.get_all_schemas()
        
        # Main agent loop
        while iterations < self.settings.max_iterations:
            iterations += 1
            
            # Get messages for API call
            messages = self.memory.get_messages()
            
            # Make API call
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.settings.max_tokens,
                )
            except Exception as e:
                error_msg = f"API error: {str(e)}"
                self.memory.add_assistant_message(error_msg)
                return AgentResponse(
                    content=error_msg,
                    iterations=iterations,
                    duration_seconds=time.time() - start_time,
                )
            
            # Track tokens
            if response.usage:
                tokens_used += response.usage.total_tokens
            
            # Get the assistant's response
            assistant_message = response.choices[0].message
            
            # Check for tool calls
            if assistant_message.tool_calls:
                # Add assistant message with tool calls to memory
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]
                
                self.memory.add_assistant_message(
                    content=assistant_message.content or "",
                    tool_calls=tool_calls_data,
                )
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # Execute the tool
                    result = self.registry.execute_tool(tool_name, arguments)
                    
                    # Format result for message
                    if isinstance(result, dict):
                        result_str = json.dumps(result, indent=2, default=str)
                    else:
                        result_str = str(result)
                    
                    # Truncate very long results
                    if len(result_str) > 10000:
                        result_str = result_str[:10000] + "\n... (truncated)"
                    
                    # Add tool result to memory
                    self.memory.add_tool_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        result=result_str,
                    )
                    
                    tool_calls_made += 1
                
                # Continue loop to process tool results
                continue
            
            # No tool calls - we have a final response
            final_content = assistant_message.content or ""
            self.memory.add_assistant_message(final_content)
            
            # Update stats
            self.total_tokens_used += tokens_used
            self.total_tool_calls += tool_calls_made
            
            return AgentResponse(
                content=final_content,
                tool_calls_made=tool_calls_made,
                tokens_used=tokens_used,
                iterations=iterations,
                duration_seconds=time.time() - start_time,
            )
        
        # Max iterations reached
        error_msg = "Maximum iterations reached. Please try a simpler request."
        self.memory.add_assistant_message(error_msg)
        
        return AgentResponse(
            content=error_msg,
            tool_calls_made=tool_calls_made,
            tokens_used=tokens_used,
            iterations=iterations,
            duration_seconds=time.time() - start_time,
        )
    
    def chat_stream(self, user_message: str) -> Generator[str, None, AgentResponse]:
        """
        Process a user message with streaming response.
        Yields content chunks as they arrive.
        
        Args:
            user_message: The user's input
            
        Yields:
            Content chunks
            
        Returns:
            AgentResponse with final stats
        """
        start_time = time.time()
        iterations = 0
        tool_calls_made = 0
        tokens_used = 0
        
        # Add user message to memory
        self.memory.add_user_message(user_message)
        
        # Get tool schemas
        tools = self.registry.get_all_schemas()
        
        # Main agent loop
        while iterations < self.settings.max_iterations:
            iterations += 1
            
            messages = self.memory.get_messages()
            
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.settings.max_tokens,
                    stream=True,
                )
            except Exception as e:
                error_msg = f"API error: {str(e)}"
                yield error_msg
                return AgentResponse(
                    content=error_msg,
                    iterations=iterations,
                    duration_seconds=time.time() - start_time,
                )
            
            # Collect streamed response
            full_content = ""
            tool_calls_buffer = {}
            
            for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Handle content
                if delta.content:
                    full_content += delta.content
                    yield delta.content
                
                # Handle tool calls (accumulate)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": "",
                                "function": {"name": "", "arguments": ""},
                            }
                        
                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buffer[idx]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_buffer[idx]["function"]["arguments"] += tc.function.arguments
            
            # Check if there were tool calls
            if tool_calls_buffer:
                # Add assistant message with tool calls
                tool_calls_data = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in tool_calls_buffer.values()
                ]
                
                self.memory.add_assistant_message(
                    content=full_content,
                    tool_calls=tool_calls_data,
                )
                
                # Execute tool calls
                for tc_data in tool_calls_buffer.values():
                    tool_name = tc_data["function"]["name"]
                    
                    yield f"\n🔧 Calling {tool_name}..."
                    
                    try:
                        arguments = json.loads(tc_data["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    result = self.registry.execute_tool(tool_name, arguments)
                    
                    if isinstance(result, dict):
                        result_str = json.dumps(result, indent=2, default=str)
                    else:
                        result_str = str(result)
                    
                    if len(result_str) > 10000:
                        result_str = result_str[:10000] + "\n... (truncated)"
                    
                    self.memory.add_tool_result(
                        tool_call_id=tc_data["id"],
                        tool_name=tool_name,
                        result=result_str,
                    )
                    
                    tool_calls_made += 1
                
                yield "\n"
                continue
            
            # No tool calls - final response
            self.memory.add_assistant_message(full_content)
            
            self.total_tokens_used += tokens_used
            self.total_tool_calls += tool_calls_made
            
            return AgentResponse(
                content=full_content,
                tool_calls_made=tool_calls_made,
                tokens_used=tokens_used,
                iterations=iterations,
                duration_seconds=time.time() - start_time,
            )
        
        error_msg = "Maximum iterations reached."
        yield error_msg
        
        return AgentResponse(
            content=error_msg,
            tool_calls_made=tool_calls_made,
            iterations=iterations,
            duration_seconds=time.time() - start_time,
        )
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self._set_system_message()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        memory_stats = self.memory.get_stats()
        
        return {
            "model": self.model,
            "temperature": self.temperature,
            "total_tokens_used": self.total_tokens_used,
            "total_tool_calls": self.total_tool_calls,
            "available_tools": len(self.registry.list_tools()),
            "memory": memory_stats,
        }


# Singleton instance
_agent: Optional[AIAgent] = None


def get_agent() -> AIAgent:
    """Get or create the AI agent"""
    global _agent
    if _agent is None:
        _agent = AIAgent()
    return _agent


def chat(message: str) -> str:
    """Simple chat function"""
    agent = get_agent()
    response = agent.chat(message)
    return response.content



