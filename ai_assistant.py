#!/usr/bin/env python3
"""
AI-Powered File System Terminal Assistant

A terminal application that uses LangChain with Gemini API to perform
file and folder operations through natural language commands.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*migration.*')

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.table import Table
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import our file system manager
from tools.file_fol_tools import FileSystemManager
from tools.terminal_manager import TerminalManager
import config

# Load environment variables
load_dotenv()

app = typer.Typer(help="AI-Powered File System Assistant")
console = Console()

class AIFileAssistant:
    """Main AI assistant for file operations"""
    
    def __init__(self, workspace_dir: str = None):
        self.console = Console()
        self.workspace_dir = Path(workspace_dir or config.WORKSPACE_DIR)
        
        # Ensure workspace directory exists
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize FileSystemManager with workspace as base directory
        self.fs_manager = FileSystemManager(base_dir=str(self.workspace_dir.resolve()))
        self.terminal_manager = TerminalManager(safe_mode=True)  # Initialize with safe mode
        
        # Initialize enhanced tools
        from tools.smart_editor import SmartFileEditor
        from tools.rag_system import WorkspaceRAG
        
        self.smart_editor = SmartFileEditor(base_dir=str(self.workspace_dir.resolve()))
        self.rag_system = WorkspaceRAG(str(self.workspace_dir.resolve()))
        
        # Index workspace on startup (silently)
        index_stats = self.rag_system.index_workspace()
        
        self.setup_llm()
        self.setup_memory()
        self.setup_tools()
        self.setup_agent()
        
    def setup_llm(self):
        """Initialize the Gemini LLM"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.console.print("❌ GOOGLE_API_KEY not found", style="red")
            self.console.print("Set in .env file: GOOGLE_API_KEY=your_key", style="yellow")
            sys.exit(1)
            
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=api_key,
            temperature=config.LLM_TEMPERATURE
        )
        
    def setup_memory(self):
        """Initialize conversation memory"""
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        
    def setup_tools(self):
        """Create LangChain tools from FileSystemManager methods"""
        self.tools = [
            # Basic file operation tools
            Tool(
                name="list_directory",
                description="List files and folders in a directory. Input: directory path (optional, defaults to current)",
                func=self._list_directory_tool
            ),
            Tool(
                name="read_file", 
                description="Read contents of a file. Input: file path (use forward slashes or just filename)",
                func=self._read_file_tool
            ),
            Tool(
                name="create_file",
                description="Create a new file with content. Input: 'filepath|content' separated by |",
                func=self._create_file_tool
            ),
            Tool(
                name="create_directory",
                description="Create a new directory. Input: directory path",
                func=self._create_directory_tool
            ),
            Tool(
                name="write_file",
                description="Write content to a file (overwrites existing). Input: 'filepath|content' separated by |",
                func=self._write_file_tool
            ),
            Tool(
                name="append_file",
                description="Append content to a file. Input: 'filepath|content' separated by |",
                func=self._append_file_tool
            ),
            Tool(
                name="delete_file",
                description="Delete a file. Input: file path",
                func=self._delete_file_tool
            ),
            
            # Enhanced editing tools
            Tool(
                name="read_file_with_lines",
                description="Read file with line numbers for editing. Input: 'filepath|start_line|end_line' (start_line and end_line optional)",
                func=self._read_file_with_lines_tool
            ),
            Tool(
                name="insert_lines",
                description="Insert content at specific line. Input: 'filepath|line_number|content'",
                func=self._insert_lines_tool
            ),
            Tool(
                name="replace_lines",
                description="Replace lines in file. Input: 'filepath|start_line|end_line|new_content'",
                func=self._replace_lines_tool
            ),
            Tool(
                name="delete_lines",
                description="Delete lines from file. Input: 'filepath|start_line|end_line' (end_line optional)",
                func=self._delete_lines_tool
            ),
            Tool(
                name="find_and_replace",
                description="Find and replace text in file. Input: 'filepath|search_text|replacement|is_regex' (is_regex optional, true/false)",
                func=self._find_replace_tool
            ),
            Tool(
                name="analyze_file_structure",
                description="Analyze file structure (functions, classes, imports). Input: file path",
                func=self._analyze_structure_tool
            ),
            
            # RAG and workspace awareness tools
            Tool(
                name="search_workspace",
                description="Semantic search across workspace using RAG. Input: search query",
                func=self._search_workspace_tool
            ),
            Tool(
                name="get_file_context",
                description="Get comprehensive context for a file using RAG. Input: file path",
                func=self._get_file_context_tool
            ),
            Tool(
                name="workspace_overview",
                description="Get overview of entire workspace. Input: none (use empty string)",
                func=self._workspace_overview_tool
            ),
            Tool(
                name="get_related_code",
                description="Find code related to a specific file. Input: file path",
                func=self._get_related_code_tool
            ),
            
            # Legacy tools (keeping for compatibility)
            Tool(
                name="search_files",
                description="Search for text in files. Input: 'search_term|directory|file_pattern' (directory and file_pattern optional)",
                func=self._search_files_tool
            ),
            Tool(
                name="analyze_code",
                description="Analyze code structure of a file. Input: file path",
                func=lambda path: json.dumps(self.fs_manager.analyze_code(path)) if path else "Error: No file path provided"
            ),
            Tool(
                name="project_context",
                description="Get overview of project structure. Input: directory path (optional)",
                func=lambda path="": json.dumps(self.fs_manager.project_context(path or "."))
            ),
            
            # Terminal operation tools
            Tool(
                name="execute_command",
                description="Execute a terminal/shell command safely. Input: command to execute",
                func=self._execute_command_tool
            ),
            Tool(
                name="get_system_info",
                description="Get comprehensive system information including OS, shell, and environment",
                func=lambda _: json.dumps(self.terminal_manager.get_system_info(), indent=2)
            ),
            Tool(
                name="suggest_commands",
                description="Suggest terminal commands based on user intent. Input: description of what you want to do",
                func=self._suggest_commands_tool
            ),
            Tool(
                name="check_command_safety",
                description="Check if a command is safe to execute. Input: command to check",
                func=self._check_command_safety_tool
            )
        ]
        
    def _create_file_tool(self, input_str: str) -> str:
        """Tool wrapper for creating files"""
        try:
            # Clean the input
            cleaned_input = input_str.strip().replace('\r', '') if input_str else ""
            if "|" not in cleaned_input:
                return "Error: Input format should be 'filepath|content'"
            filepath, content = cleaned_input.split("|", 1)
            # Clean the filepath but preserve content formatting
            cleaned_filepath = filepath.strip().replace('\n', '')
            result = self.fs_manager.file_creator(cleaned_filepath, content=content, debug=True)
            return f"✅ File created: {result}"
        except Exception as e:
            return f"❌ Error creating file: {str(e)}"
    
    def _create_directory_tool(self, dirpath: str) -> str:
        """Tool wrapper for creating directories"""
        try:
            # Clean the input by stripping all whitespace including newlines
            cleaned_path = dirpath.strip().replace('\n', '').replace('\r', '') if dirpath else ""
            if not cleaned_path:
                return "Error: No directory path provided"
            result = self.fs_manager.dir_creator(cleaned_path, debug=True)
            return f"✅ Directory created: {result}"
        except Exception as e:
            return f"❌ Error creating directory: {str(e)}"
            
    def _write_file_tool(self, input_str: str) -> str:
        """Tool wrapper for writing files"""
        try:
            if "|" not in input_str:
                return "Error: Input format should be 'filepath|content'"
            filepath, content = input_str.split("|", 1)
            result = self.fs_manager.writer_func(filepath.strip(), "write", content=content)
            return f" {result.get('message', 'File written successfully')}"
        except Exception as e:
            return f"❌ Error writing file: {str(e)}"
            
    def _append_file_tool(self, input_str: str) -> str:
        """Tool wrapper for appending to files"""
        try:
            if "|" not in input_str:
                return "Error: Input format should be 'filepath|content'"
            filepath, content = input_str.split("|", 1)
            result = self.fs_manager.writer_func(filepath.strip(), "append", content=content)
            return f"✅ {result.get('message', 'Content appended successfully')}"
        except Exception as e:
            return f"❌ Error appending to file: {str(e)}"
            
    def _delete_file_tool(self, filepath: str) -> str:
        """Tool wrapper for deleting files"""
        try:
            result = self.fs_manager.writer_func(filepath.strip(), "delete_file")
            return f"✅ {result.get('message', 'File deleted successfully')}"
        except Exception as e:
            return f"❌ Error deleting file: {str(e)}"
            
    def _search_files_tool(self, input_str: str) -> str:
        """Tool wrapper for searching in files"""
        try:
            parts = input_str.split("|")
            search_term = parts[0].strip()
            directory = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "."
            file_pattern = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "*.*"
            
            result = self.fs_manager.search_in_files(search_term, directory, file_pattern)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f" Error searching files: {str(e)}"
    
    # Terminal operation tool wrappers
    def _execute_command_tool(self, command: str) -> str:
        """Tool wrapper for executing terminal commands"""
        try:
            if not command or not command.strip():
                return "Error: No command provided"
            
            result = self.terminal_manager.execute_command(command.strip())
            
            if result.get('success'):
                output = f" Command executed successfully:\n"
                output += f"Command: {result['command']}\n"
                output += f"Return code: {result['return_code']}\n"
                output += f"Execution time: {result['execution_time']:.2f}s\n"
                
                if result['stdout']:
                    output += f"Output:\n{result['stdout']}\n"
                if result['stderr']:
                    output += f"Errors/Warnings:\n{result['stderr']}\n"
                    
                return output
            else:
                error_msg = f" Command failed:\n"
                error_msg += f"Command: {result['command']}\n"
                if 'error' in result:
                    error_msg += f"Error: {result['error']}\n"
                if result.get('stderr'):
                    error_msg += f"stderr: {result['stderr']}\n"
                if result.get('return_code', -1) != -1:
                    error_msg += f"Return code: {result['return_code']}\n"
                return error_msg
                
        except Exception as e:
            return f" Error executing command: {str(e)}"
    
    def _suggest_commands_tool(self, intent: str) -> str:
        """Tool wrapper for suggesting commands"""
        try:
            suggestions = self.terminal_manager.suggest_command(intent)
            if suggestions:
                output = f"💡 Suggested commands for '{intent}':\n"
                for i, cmd in enumerate(suggestions, 1):
                    output += f"{i}. {cmd}\n"
                return output
            else:
                return f"No specific suggestions found for '{intent}'. Try being more specific about what you want to do."
        except Exception as e:
            return f" Error suggesting commands: {str(e)}"
    
    def _check_command_safety_tool(self, command: str) -> str:
        """Tool wrapper for checking command safety"""
        try:
            is_safe, reason = self.terminal_manager.is_command_safe(command)
            if is_safe:
                return f"✅ Command is SAFE: {reason}"
            else:
                return f"⚠️ Command is BLOCKED: {reason}"
        except Exception as e:
            return f"❌ Error checking command safety: {str(e)}"
    
    def _list_directory_tool(self, path: str = "") -> str:
        """Tool wrapper for listing directory contents - returns human readable format"""
        try:
            structure = self.fs_manager.folder_structure(path or ".")
            
            # Convert nested structure to readable format
            def format_structure(struct, indent=0):
                result = []
                for key, value in struct.items():
                    if key == '_files':
                        for file in value:
                            result.append("  " * indent + f"📄 {file}")
                    else:
                        result.append("  " * indent + f"📁 {key}/")
                        if isinstance(value, dict):
                            result.extend(format_structure(value, indent + 1))
                return result
            
            formatted_lines = format_structure(structure)
            if not formatted_lines:
                return "📁 Directory is empty"
            
            return "📁 Workspace Contents:\n" + "\n".join(formatted_lines)
            
        except Exception as e:
            return f"❌ Error listing directory: {str(e)}"
    
    def _read_file_tool(self, path: str) -> str:
        """Tool wrapper for reading file contents with better error handling"""
        try:
            if not path:
                return "❌ Error: No file path provided"
            
            # Clean the path and handle different formats
            clean_path = path.strip().replace('\\', '/')
            content = self.fs_manager.reader_func(clean_path)
            
            # Ensure string return type
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            
            return str(content)
            
        except Exception as e:
            return f"❌ Error reading file '{path}': {str(e)}"
    
    # Enhanced editing tool wrappers
    def _read_file_with_lines_tool(self, input_str: str) -> str:
        """Tool wrapper for reading file with line numbers"""
        try:
            parts = input_str.split('|')
            filepath = parts[0]
            start_line = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 1
            end_line = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
            
            result = self.smart_editor.read_file_with_lines(filepath, start_line, end_line)
            if result['success']:
                formatted_lines = '\n'.join(result['lines_with_numbers'])
                return f"📄 {result['file_path']} (lines {result['start_line']}-{result['end_line']}):\n{formatted_lines}"
            else:
                return f"❌ Error: {result['error']}"
        except Exception as e:
            return f"❌ Error reading file with lines: {str(e)}"
    
    def _insert_lines_tool(self, input_str: str) -> str:
        """Tool wrapper for inserting lines"""
        try:
            if input_str.count('|') < 2:
                return "Error: Input format should be 'filepath|line_number|content'"
            parts = input_str.split('|', 2)
            filepath, line_num, content = parts[0], int(parts[1]), parts[2]
            
            result = self.smart_editor.insert_lines(filepath, line_num, content)
            if result['success']:
                return f"✅ {result['message']}"
            else:
                return f"❌ Error: {result['error']}"
        except Exception as e:
            return f"❌ Error inserting lines: {str(e)}"
    
    def _replace_lines_tool(self, input_str: str) -> str:
        """Tool wrapper for replacing lines"""
        try:
            if input_str.count('|') < 3:
                return "Error: Input format should be 'filepath|start_line|end_line|content'"
            parts = input_str.split('|', 3)
            filepath, start_line, end_line, content = parts[0], int(parts[1]), int(parts[2]), parts[3]
            
            result = self.smart_editor.replace_lines(filepath, start_line, end_line, content)
            if result['success']:
                return f"✅ {result['message']}"
            else:
                return f"❌ Error: {result['error']}"
        except Exception as e:
            return f"❌ Error replacing lines: {str(e)}"
    
    def _delete_lines_tool(self, input_str: str) -> str:
        """Tool wrapper for deleting lines"""
        try:
            parts = input_str.split('|')
            filepath = parts[0]
            start_line = int(parts[1]) if len(parts) > 1 else 1
            end_line = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
            
            result = self.smart_editor.delete_lines(filepath, start_line, end_line)
            if result['success']:
                return f"✅ {result['message']}"
            else:
                return f"❌ Error: {result['error']}"
        except Exception as e:
            return f"❌ Error deleting lines: {str(e)}"
    
    def _find_replace_tool(self, input_str: str) -> str:
        """Tool wrapper for find and replace"""
        try:
            if input_str.count('|') < 2:
                return "Error: Input format should be 'filepath|search_text|replacement|is_regex'"
            parts = input_str.split('|', 3)
            filepath, search_text, replacement = parts[0], parts[1], parts[2]
            is_regex = parts[3].lower() == 'true' if len(parts) > 3 else False
            
            result = self.smart_editor.find_and_replace(filepath, search_text, replacement, is_regex)
            if result['success']:
                return f"✅ {result['message']}"
            else:
                return f"❌ Error: {result['error']}"
        except Exception as e:
            return f"❌ Error in find/replace: {str(e)}"
    
    def _analyze_structure_tool(self, filepath: str) -> str:
        """Tool wrapper for analyzing file structure"""
        try:
            result = self.smart_editor.analyze_file_structure(filepath)
            if result['success']:
                return f"📋 Structure of {result['file_path']} ({result['language']}):\n" + \
                       f"Functions: {len(result['functions'])}\n" + \
                       f"Classes: {len(result['classes'])}\n" + \
                       f"Structure: {json.dumps(result['structure'], indent=2)}"
            else:
                return f"❌ Error: {result['error']}"
        except Exception as e:
            return f"❌ Error analyzing structure: {str(e)}"
    
    # RAG tool wrappers
    def _search_workspace_tool(self, query: str) -> str:
        """Tool wrapper for workspace semantic search"""
        try:
            results = self.rag_system.search_code(query, limit=5)
            if results:
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        f"📄 {result['file_path']} ({result['chunk_type']}, lines {result['start_line']}-{result['end_line']}):\n" +
                        f"{result['content'][:200]}{'...' if len(result['content']) > 200 else ''}\n"
                    )
                return "🔍 Search results:\n" + "\n".join(formatted_results)
            else:
                return f"No results found for '{query}'"
        except Exception as e:
            return f"❌ Error in workspace search: {str(e)}"
    
    def _get_file_context_tool(self, filepath: str) -> str:
        """Tool wrapper for getting file context"""
        try:
            context = self.rag_system.get_file_context(filepath)
            if 'error' in context:
                return f"❌ {context['error']}"
            
            return f"📄 Context for {context['file_path']} ({context['language']}):\n" + \
                   f"Functions: {len(context['functions'])}\n" + \
                   f"Classes: {len(context['classes'])}\n" + \
                   f"Structure: {json.dumps(context['structure'], indent=2)}"
        except Exception as e:
            return f"❌ Error getting file context: {str(e)}"
    
    def _workspace_overview_tool(self, _: str) -> str:
        """Tool wrapper for workspace overview"""
        try:
            overview = self.rag_system.get_workspace_overview()
            return f"📊 Workspace Overview:\n" + \
                   f"Files: {overview['total_files']}\n" + \
                   f"Code chunks: {overview['total_chunks']}\n" + \
                   f"Languages: {json.dumps(overview['languages'], indent=2)}\n" + \
                   f"Recent files: {overview['recent_files'][:5]}"
        except Exception as e:
            return f"❌ Error getting workspace overview: {str(e)}"
    
    def _get_related_code_tool(self, filepath: str) -> str:
        """Tool wrapper for getting related code"""
        try:
            related = self.rag_system.get_related_code(filepath)
            if related:
                formatted_results = []
                for result in related[:3]:  # Limit to top 3
                    formatted_results.append(
                        f"📄 {result['file_path']} ({result['chunk_type']}):\n" +
                        f"{result['content'][:150]}{'...' if len(result['content']) > 150 else ''}"
                    )
                return f"🔗 Related code for {filepath}:\n" + "\n\n".join(formatted_results)
            else:
                return f"No related code found for {filepath}"
        except Exception as e:
            return f"❌ Error finding related code: {str(e)}"
            
    def setup_agent(self):
        """Create the LangChain agent"""
        
        # Create the prompt template
        workspace_path = self.workspace_dir.resolve()
        prompt_template = f"""You are an AI assistant specialized in file, folder, and terminal operations. You can help users manage their files and directories, and execute terminal commands through natural language commands.

IMPORTANT: You are working in a SANDBOXED WORKSPACE directory at: {workspace_path}
All file operations are restricted to this workspace to protect the application's source code.
When creating files or directories, you must use paths relative to the workspace directory or use ./filename for files in the workspace root.

CAPABILITIES:
- File Operations: Create, read, write, delete, search files and directories (within workspace only)
- Terminal Operations: Execute shell commands safely (with safety restrictions)
- System Information: Get OS details, shell info, environment variables
- Cross-platform: Works on Windows (CMD/PowerShell), Linux, and macOS

SAFETY RULES:
- All file operations are restricted to the workspace directory
- Terminal commands run in SAFE MODE by default
- Only allow safe, non-destructive commands
- Block dangerous operations like system shutdown, formatting, etc.
- Always explain what a command does before executing

INPUT FORMAT RULES:
- For create_directory: Use relative paths (e.g., "subfolder" or "documents/projects") 
- For create_file: Use "relative_path/filename|content" format (e.g., "test.txt|content" or "documents/note.txt|content")
- For execute_command: Action Input should be just the command without extra formatting
- DO NOT include newlines, quotes, or extra formatting in Action Input
- All file paths are automatically resolved within the workspace directory

Available tools:
{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action (follow format rules above - no newlines or extra formatting)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Chat History:
{{chat_history}}

Question: {{input}}
{{agent_scratchpad}}"""

        prompt = PromptTemplate(
            input_variables=["input", "intermediate_steps", "tools", "tool_names", "chat_history", "agent_scratchpad"],
            template=prompt_template
        )
        
        # Create the agent
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        # Create agent executor with cleaner output
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,  # Hide internal reasoning for cleaner output
            handle_parsing_errors=True,
            max_iterations=15,  # Increased from 5 to prevent timeouts
            return_intermediate_steps=False  # Don't return internal steps
        )
        
    def process_command(self, user_input: str) -> str:
        """Process user command through the AI agent"""
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "No response generated")
        except Exception as e:
            return f" Error processing command: {str(e)}"
            
    def run_interactive(self):
        """Run the interactive terminal session"""
        self.console.print(Panel.fit(
            "🤖 AI Assistant",
            subtitle="Gemini Flash + RAG",
            style="bold orange1",
            border_style="orange1"
        ))
        
        self.console.print(f"📁 [orange3]{self.workspace_dir.resolve()}[/orange3]")
        self.console.print("🔒 [dim orange3]Sandboxed workspace[/dim orange3]")
        
        self.console.print("\n💡 [orange3]Commands:[/orange3]")
        examples = [
            "• List files",
            "• Create file 'notes.txt'", 
            "• Read README.md",
            "• Search for 'TODO'",
            "• What's in app.py?"
        ]
        
        for example in examples:
            self.console.print(f"  [dim]{example}[/dim]")
            
        self.console.print("\n📝 [dim]Type 'quit' to exit[/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold orange1]You[/bold orange1]")
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    self.console.print("👋 Goodbye!", style="orange1")
                    break
                    
                if not user_input.strip():
                    continue
                    
                self.console.print("\n[bold blue]🤖 Assistant[/bold blue]")
                
                # Show thinking indicator
                with self.console.status("[bold green]Thinking..."):
                    response = self.process_command(user_input)
                    
                self.console.print(response)
                
            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!", style="green")
                break
            except EOFError:
                # Handle EOF when input is piped or redirected
                self.console.print("\n👋 Goodbye!", style="green")
                break
            except Exception as e:
                self.console.print(f"❌ Unexpected error: {str(e)}", style="red")
                # Break on repeated errors to prevent infinite loops
                break


@app.command()
def interactive():
    """Start the interactive AI assistant"""
    assistant = AIFileAssistant()
    assistant.run_interactive()


@app.command() 
def execute(
    command: str = typer.Argument(..., help="Natural language command to execute"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Execute a single command and exit"""
    console = Console()
    
    if verbose:
        console.print("🤖 AI in CLI - Direct Command Mode", style="bold orange1")
        console.print(f"📝 Command: {command}", style="dim")
        console.print()
    
    try:
        assistant = AIFileAssistant()
        
        if verbose:
            with console.status("[bold orange1]Processing..."):
                response = assistant.process_command(command)
        else:
            response = assistant.process_command(command)
            
        console.print("🤖", response, style="green")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def setup():
    """Setup the environment and API keys"""
    console.print("🔧 Setting up AI File Assistant...", style="blue")
    
    # Check if .env file exists
    env_path = Path(".env")
    if not env_path.exists():
        console.print("📝 Creating .env file...", style="yellow")
        api_key = Prompt.ask("Enter your Google API Key")
        
        with open(".env", "w") as f:
            f.write(f"GOOGLE_API_KEY={api_key}\n")
            
        console.print(" .env file created!", style="green")
    else:
        console.print(" .env file already exists", style="green")
        
    # Test API connection
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key
            )
            response = llm.invoke([HumanMessage(content="Say 'API connection successful'")])
            console.print(" API connection test successful!", style="green")
        else:
            console.print(" No API key found", style="red")
    except Exception as e:
        console.print(f" API connection failed: {str(e)}", style="red")


@app.command()
def list_tools():
    """List all available tools and their descriptions"""
    assistant = AIFileAssistant()
    
    table = Table(title="Available Tools")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="white")
    
    for tool in assistant.tools:
        table.add_row(tool.name, tool.description)
        
    console.print(table)


if __name__ == "__main__":
    app()
