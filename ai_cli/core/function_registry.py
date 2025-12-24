"""
Function Registry - Tool definitions for OpenAI function calling
Defines all available tools and their schemas
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of a tool"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function schema"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            
            if param.enum:
                prop["enum"] = param.enum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class FunctionRegistry:
    """
    Registry of all available tools for the AI agent.
    Manages tool definitions and execution.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all available tools"""
        # File Tools
        self._register_file_tools()
        
        # Folder Tools
        self._register_folder_tools()
        
        # Code Tools
        self._register_code_tools()
        
        # Terminal Tools
        self._register_terminal_tools()
        
        # Search Tools
        self._register_search_tools()
    
    def _register_file_tools(self):
        """Register file operation tools"""
        from tools.file_tools import FileTools
        file_tools = FileTools()
        
        # Read file
        self.register(ToolDefinition(
            name="read_file",
            description="Read the contents of a file. Returns the file content, line count, and metadata.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file (relative to workspace)",
                ),
            ],
            handler=lambda filepath: file_tools.read_file(filepath),
        ))
        
        # Read file with line numbers
        self.register(ToolDefinition(
            name="read_file_lines",
            description="Read specific lines from a file with line numbers. Useful for viewing sections of code.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="start_line",
                    type="integer",
                    description="Starting line number (1-indexed)",
                    required=False,
                    default=1,
                ),
                ToolParameter(
                    name="end_line",
                    type="integer",
                    description="Ending line number (1-indexed, inclusive). Omit for end of file.",
                    required=False,
                ),
            ],
            handler=lambda filepath, start_line=1, end_line=None: file_tools.read_file_lines(filepath, start_line, end_line),
        ))
        
        # Create file
        self.register(ToolDefinition(
            name="create_file",
            description="Create a new file with content. Creates parent directories if needed.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path for the new file",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=False,
                    default="",
                ),
            ],
            handler=lambda filepath, content="": file_tools.create_file(filepath, content),
        ))
        
        # Write file
        self.register(ToolDefinition(
            name="write_file",
            description="Write/overwrite content to a file. Use for complete file replacement.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write",
                ),
            ],
            handler=lambda filepath, content: file_tools.write_file(filepath, content),
        ))
        
        # Delete file
        self.register(ToolDefinition(
            name="delete_file",
            description="Delete a file from the workspace.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file to delete",
                ),
            ],
            handler=lambda filepath: file_tools.delete_file(filepath),
        ))
        
        # Get file info
        self.register(ToolDefinition(
            name="get_file_info",
            description="Get detailed information about a file (size, dates, permissions).",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
            ],
            handler=lambda filepath: file_tools.get_file_info(filepath),
        ))
    
    def _register_folder_tools(self):
        """Register folder operation tools"""
        from tools.folder_tools import FolderTools
        folder_tools = FolderTools()
        
        # List directory
        self.register(ToolDefinition(
            name="list_directory",
            description="List contents of a directory. Shows files and folders.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path (default: current workspace)",
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="If true, list recursively as a tree",
                    required=False,
                    default=False,
                ),
            ],
            handler=lambda path=".", recursive=False: folder_tools.list_directory(path, recursive),
        ))
        
        # Create directory
        self.register(ToolDefinition(
            name="create_directory",
            description="Create a new directory. Creates parent directories if needed.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path to create",
                ),
            ],
            handler=lambda path: folder_tools.create_directory(path),
        ))
        
        # Delete directory
        self.register(ToolDefinition(
            name="delete_directory",
            description="Delete a directory. Use recursive=true to delete non-empty directories.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path to delete",
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="If true, delete all contents recursively",
                    required=False,
                    default=False,
                ),
            ],
            handler=lambda path, recursive=False: folder_tools.delete_directory(path, recursive),
        ))
        
        # Find files
        self.register(ToolDefinition(
            name="find_files",
            description="Find files matching a glob pattern (e.g., '*.py', '*test*').",
            parameters=[
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="Glob pattern to match",
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Starting directory",
                    required=False,
                    default=".",
                ),
            ],
            handler=lambda pattern, path=".": folder_tools.find_files(pattern, path),
        ))
        
        # Get directory info
        self.register(ToolDefinition(
            name="get_directory_info",
            description="Get statistics about a directory (file count, size, types).",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path",
                    required=False,
                    default=".",
                ),
            ],
            handler=lambda path=".": folder_tools.get_directory_info(path),
        ))
    
    def _register_code_tools(self):
        """Register code editing tools"""
        from tools.code_tools import CodeTools
        code_tools = CodeTools()
        
        # Insert lines
        self.register(ToolDefinition(
            name="insert_lines",
            description="Insert content at a specific line number in a file.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="line_number",
                    type="integer",
                    description="Line number to insert at (1-indexed)",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to insert",
                ),
            ],
            handler=lambda filepath, line_number, content: code_tools.insert_lines(filepath, line_number, content),
        ))
        
        # Replace lines
        self.register(ToolDefinition(
            name="replace_lines",
            description="Replace a range of lines with new content.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="start_line",
                    type="integer",
                    description="Starting line number (1-indexed, inclusive)",
                ),
                ToolParameter(
                    name="end_line",
                    type="integer",
                    description="Ending line number (1-indexed, inclusive)",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Replacement content",
                ),
            ],
            handler=lambda filepath, start_line, end_line, content: code_tools.replace_lines(filepath, start_line, end_line, content),
        ))
        
        # Delete lines
        self.register(ToolDefinition(
            name="delete_lines",
            description="Delete a range of lines from a file.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="start_line",
                    type="integer",
                    description="Starting line number (1-indexed, inclusive)",
                ),
                ToolParameter(
                    name="end_line",
                    type="integer",
                    description="Ending line number (1-indexed, inclusive)",
                ),
            ],
            handler=lambda filepath, start_line, end_line: code_tools.delete_lines(filepath, start_line, end_line),
        ))
        
        # Find and replace
        self.register(ToolDefinition(
            name="find_and_replace",
            description="Find and replace text in a file. Supports regex patterns.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="find",
                    type="string",
                    description="Text or pattern to find",
                ),
                ToolParameter(
                    name="replace",
                    type="string",
                    description="Replacement text",
                ),
                ToolParameter(
                    name="regex",
                    type="boolean",
                    description="Treat find as regex pattern",
                    required=False,
                    default=False,
                ),
            ],
            handler=lambda filepath, find, replace, regex=False: code_tools.find_and_replace(filepath, find, replace, regex),
        ))
        
        # Analyze code
        self.register(ToolDefinition(
            name="analyze_code",
            description="Analyze Python code structure (classes, functions, imports).",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the Python file",
                ),
            ],
            handler=lambda filepath: code_tools.analyze_code(filepath),
        ))
        
        # Apply edit
        self.register(ToolDefinition(
            name="apply_edit",
            description="Apply a precise edit by finding and replacing exact content.",
            parameters=[
                ToolParameter(
                    name="filepath",
                    type="string",
                    description="Path to the file",
                ),
                ToolParameter(
                    name="old_content",
                    type="string",
                    description="Exact content to find (must be unique)",
                ),
                ToolParameter(
                    name="new_content",
                    type="string",
                    description="Content to replace with",
                ),
            ],
            handler=lambda filepath, old_content, new_content: code_tools.apply_edit(filepath, old_content, new_content),
        ))
    
    def _register_terminal_tools(self):
        """Register terminal/command tools"""
        from tools.terminal_tools import TerminalTools
        terminal_tools = TerminalTools()
        
        # Execute command
        self.register(ToolDefinition(
            name="execute_command",
            description="Execute a terminal command in the sandbox. Commands are validated for safety.",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="Command to execute",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds",
                    required=False,
                    default=30,
                ),
            ],
            handler=lambda command, timeout=30: terminal_tools.execute_command(command, timeout),
        ))
        
        # Run Python code
        self.register(ToolDefinition(
            name="run_python_code",
            description="Execute Python code directly.",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds",
                    required=False,
                    default=60,
                ),
            ],
            handler=lambda code, timeout=60: terminal_tools.run_python(code=code, timeout=timeout),
        ))
        
        # Run Python script
        self.register(ToolDefinition(
            name="run_python_script",
            description="Execute a Python script file.",
            parameters=[
                ToolParameter(
                    name="script_path",
                    type="string",
                    description="Path to the Python script",
                ),
                ToolParameter(
                    name="args",
                    type="string",
                    description="Command line arguments",
                    required=False,
                    default="",
                ),
            ],
            handler=lambda script_path, args="": terminal_tools.run_python(script_path=script_path, args=args.split() if args else []),
        ))
        
        # Get system info
        self.register(ToolDefinition(
            name="get_system_info",
            description="Get system information (OS, Python version, etc.).",
            parameters=[],
            handler=lambda: terminal_tools.get_system_info(),
        ))
    
    def _register_search_tools(self):
        """Register search tools"""
        from tools.search_tools import SearchTools
        search_tools = SearchTools()
        
        # Search codebase
        self.register(ToolDefinition(
            name="search_codebase",
            description="Search the codebase using semantic, keyword, and graph-based retrieval.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                ),
                ToolParameter(
                    name="n_results",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=10,
                ),
            ],
            handler=lambda query, n_results=10: search_tools.search(query, n_results),
        ))
        
        # Search symbol
        self.register(ToolDefinition(
            name="search_symbol",
            description="Search for a code symbol (function, class, variable) by name.",
            parameters=[
                ToolParameter(
                    name="symbol_name",
                    type="string",
                    description="Symbol name to search for",
                ),
            ],
            handler=lambda symbol_name: search_tools.search_symbol(symbol_name),
        ))
        
        # Get workspace overview
        self.register(ToolDefinition(
            name="get_workspace_overview",
            description="Get an overview of the workspace with statistics.",
            parameters=[],
            handler=lambda: search_tools.get_workspace_overview(),
        ))
        
        # Index workspace
        self.register(ToolDefinition(
            name="index_workspace",
            description="Index the workspace for search. Run this to enable semantic search.",
            parameters=[
                ToolParameter(
                    name="force",
                    type="boolean",
                    description="If true, re-index everything",
                    required=False,
                    default=False,
                ),
            ],
            handler=lambda force=False: search_tools.index_workspace(force),
        ))
    
    def register(self, tool: ToolDefinition):
        """Register a tool"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI schemas for all tools"""
        return [tool.to_openai_schema() for tool in self.tools.values()]
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments"""
        tool = self.tools.get(name)
        
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        
        if not tool.handler:
            return {"error": f"Tool {name} has no handler"}
        
        try:
            result = tool.handler(**arguments)
            return result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())


# Singleton instance
_registry: Optional[FunctionRegistry] = None


def get_registry() -> FunctionRegistry:
    """Get or create the function registry"""
    global _registry
    if _registry is None:
        _registry = FunctionRegistry()
    return _registry



