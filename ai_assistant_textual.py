#!/usr/bin/env python3
"""
🤖 AI in CLI - Beautiful AI-Powered File Assistant

A stunning terminal application with RAG capabilities, smart file editing,
and workspace awareness powered by Google Gemini AI.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Core imports
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# UI imports
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Input, RichLog, Static, Button, 
    DirectoryTree, TabbedContent, TabPane, Label, 
    LoadingIndicator, ProgressBar
)
from textual.reactive import reactive
from textual.message import Message
from textual import on
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich.table import Table
from rich.markdown import Markdown

# Local imports
import config
from tools.file_fol_tools import FileSystemManager
from tools.terminal_manager import TerminalManager

# Load environment variables
load_dotenv()

console = Console()


class RAGSystem:
    """RAG (Retrieval Augmented Generation) system for workspace understanding"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chroma_client = None
        self.collection = None
        self.setup_rag()
    
    def setup_rag(self):
        """Initialize RAG components"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.workspace_dir / ".rag_db")
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("workspace_docs")
            except:
                self.collection = self.chroma_client.create_collection("workspace_docs")
                
            console.print("✅ RAG system initialized", style="green")
            
        except Exception as e:
            console.print(f"⚠️ RAG initialization failed: {e}", style="yellow")
    
    def index_workspace(self):
        """Index all files in the workspace"""
        if not self.collection:
            return {"status": "error", "message": "RAG not initialized"}
        
        indexed_files = 0
        skipped_files = 0
        
        try:
            # Clear existing documents
            self.collection.delete(where={})
            
            # Process all text files in workspace
            for file_path in self.workspace_dir.rglob("*"):
                if file_path.is_file() and self._is_text_file(file_path):
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        
                        # Split content into chunks
                        chunks = self.text_splitter.split_text(content)
                        
                        # Add chunks to vector store
                        for i, chunk in enumerate(chunks):
                            doc_id = f"{file_path.name}_{i}"
                            metadata = {
                                "source": str(file_path.relative_to(self.workspace_dir)),
                                "file_name": file_path.name,
                                "chunk_index": i,
                                "file_type": file_path.suffix,
                                "modified": str(file_path.stat().st_mtime)
                            }
                            
                            self.collection.add(
                                documents=[chunk],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
                        
                        indexed_files += 1
                        
                    except Exception as e:
                        console.print(f"⚠️ Failed to index {file_path}: {e}", style="yellow")
                        skipped_files += 1
            
            return {
                "status": "success",
                "indexed_files": indexed_files,
                "skipped_files": skipped_files,
                "message": f"Indexed {indexed_files} files, skipped {skipped_files}"
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Indexing failed: {e}"}
    
    def search_workspace(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search the workspace for relevant content"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
        except Exception as e:
            console.print(f"⚠️ Search failed: {e}", style="yellow")
            return []
    
    def get_workspace_context(self, query: str) -> str:
        """Get relevant workspace context for a query"""
        results = self.search_workspace(query)
        
        if not results:
            return "No relevant workspace content found."
        
        context_parts = []
        for result in results:
            source = result['metadata']['source']
            content = result['content']
            context_parts.append(f"From {source}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file that should be indexed"""
        text_extensions = {
            '.py', '.js', '.ts', '.html', '.css', '.md', '.txt', '.json',
            '.yaml', '.yml', '.xml', '.csv', '.sql', '.sh', '.bat', '.ps1',
            '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.php',
            '.rb', '.scala', '.kt', '.swift', '.r', '.m', '.pl', '.lua'
        }
        
        # Skip hidden files and directories
        if file_path.name.startswith('.'):
            return False
            
        # Skip files that are too large (>1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except:
            return False
        
        return file_path.suffix.lower() in text_extensions


class AIAssistantWidget(Static):
    """Main AI assistant widget"""
    
    def __init__(self):
        super().__init__()
        self.fs_manager = None
        self.terminal_manager = None
        self.llm = None
        self.memory = None
        self.agent_executor = None
        self.rag_system = None
        self.workspace_dir = None
        
    async def setup_ai(self):
        """Initialize AI components"""
        try:
            # Setup workspace
            self.workspace_dir = Path(config.get_workspace_path())
            self.workspace_dir.mkdir(exist_ok=True)
            
            # Initialize RAG system
            self.rag_system = RAGSystem(str(self.workspace_dir))
            
            # Initialize file system manager
            self.fs_manager = FileSystemManager(base_dir=str(self.workspace_dir.resolve()))
            
            # Initialize terminal manager
            self.terminal_manager = TerminalManager(safe_mode=True)
            
            # Setup LLM
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
                
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1
            )
            
            # Setup memory
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10
            )
            
            # Setup tools
            tools = self._create_tools()
            
            # Setup agent
            self.agent_executor = self._create_agent(tools)
            
            return True
            
        except Exception as e:
            console.print(f"❌ AI setup failed: {e}", style="red")
            return False
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="create_file",
                description="Create a new file with content. Input format: 'filepath|content'",
                func=self._create_file_tool
            ),
            Tool(
                name="read_file",
                description="Read the contents of a file. Input: filepath",
                func=self._read_file_tool
            ),
            Tool(
                name="edit_file",
                description="Edit a file by replacing specific text. Input format: 'filepath|old_text|new_text'",
                func=self._edit_file_tool
            ),
            Tool(
                name="list_directory",
                description="List contents of a directory. Input: directory_path (optional, defaults to current)",
                func=self._list_directory_tool
            ),
            Tool(
                name="search_workspace",
                description="Search for content in the workspace using RAG. Input: search_query",
                func=self._search_workspace_tool
            ),
            Tool(
                name="get_workspace_context",
                description="Get relevant context from workspace for a query. Input: query",
                func=self._get_context_tool
            ),
            Tool(
                name="execute_command",
                description="Execute a safe terminal command. Input: command",
                func=self._execute_command_tool
            ),
            Tool(
                name="index_workspace",
                description="Re-index all files in the workspace for better search. No input required.",
                func=self._index_workspace_tool
            )
        ]
    
    def _create_agent(self, tools: List[Tool]) -> AgentExecutor:
        """Create the main AI agent"""
        workspace_path = self.workspace_dir.resolve()
        
        prompt_template = f"""You are an advanced AI file assistant with RAG capabilities working in a sandboxed workspace at: {workspace_path}

You have access to:
- File operations (create, read, edit, list)
- Terminal command execution (safe mode)
- RAG-powered workspace search and context retrieval
- Workspace indexing for better understanding

IMPORTANT RULES:
- ALL file operations are restricted to the workspace directory
- Use RAG search to understand existing code before making changes
- Always get workspace context before editing files
- Prefer precise edits over full file rewrites
- Use orange-themed responses and be helpful

Available tools:
{{tools}}

Use this format:
Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
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
        
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    # Tool implementations
    def _create_file_tool(self, input_str: str) -> str:
        try:
            if "|" not in input_str:
                return "Error: Input format should be 'filepath|content'"
            filepath, content = input_str.split("|", 1)
            filepath = filepath.strip()
            result = self.fs_manager.file_creator(filepath, content=content)
            
            # Re-index after file creation
            if self.rag_system:
                asyncio.create_task(self._reindex_background())
            
            return f"✅ File created: {result}"
        except Exception as e:
            return f"❌ Error creating file: {str(e)}"
    
    def _read_file_tool(self, filepath: str) -> str:
        try:
            content = self.fs_manager.reader_func(filepath.strip())
            return f"📄 Content of {filepath}:\n{content}"
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
    
    def _edit_file_tool(self, input_str: str) -> str:
        try:
            parts = input_str.split("|")
            if len(parts) != 3:
                return "Error: Input format should be 'filepath|old_text|new_text'"
            
            filepath, old_text, new_text = parts
            filepath = filepath.strip()
            old_text = old_text.strip()
            new_text = new_text.strip()
            
            # Read current content
            current_content = self.fs_manager.reader_func(filepath)
            
            # Replace text
            if old_text in current_content:
                new_content = current_content.replace(old_text, new_text)
                self.fs_manager.writer_func(filepath, "write", content=new_content)
                
                # Re-index after editing
                if self.rag_system:
                    asyncio.create_task(self._reindex_background())
                
                return f"✅ File edited: {filepath}"
            else:
                return f"❌ Text '{old_text}' not found in {filepath}"
                
        except Exception as e:
            return f"❌ Error editing file: {str(e)}"
    
    def _list_directory_tool(self, directory_path: str = ".") -> str:
        try:
            result = self.fs_manager.folder_structure(directory_path.strip() or ".")
            
            def format_structure(structure, indent=0):
                output = []
                for key, value in structure.items():
                    if key == '_files':
                        for file in value:
                            output.append("  " * indent + f"📄 {file}")
                    else:
                        output.append("  " * indent + f"📁 {key}/")
                        if isinstance(value, dict):
                            output.extend(format_structure(value, indent + 1))
                return output
            
            structure_lines = format_structure(result)
            return f"📁 Directory structure:\n" + "\n".join(structure_lines)
            
        except Exception as e:
            return f"❌ Error listing directory: {str(e)}"
    
    def _search_workspace_tool(self, query: str) -> str:
        try:
            if not self.rag_system:
                return "❌ RAG system not available"
            
            results = self.rag_system.search_workspace(query.strip())
            
            if not results:
                return f"🔍 No results found for: {query}"
            
            output = [f"🔍 Search results for '{query}':"]
            for i, result in enumerate(results[:3], 1):
                source = result['metadata']['source']
                content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                output.append(f"\n{i}. From {source}:\n{content}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    def _get_context_tool(self, query: str) -> str:
        try:
            if not self.rag_system:
                return "❌ RAG system not available"
            
            context = self.rag_system.get_workspace_context(query.strip())
            return f"🧠 Workspace context for '{query}':\n{context}"
            
        except Exception as e:
            return f"❌ Context retrieval error: {str(e)}"
    
    def _execute_command_tool(self, command: str) -> str:
        try:
            result = self.terminal_manager.execute_command(command.strip())
            
            if result.get('success'):
                output = f"✅ Command executed successfully:\n"
                output += f"Command: {result['command']}\n"
                if result['stdout']:
                    output += f"Output:\n{result['stdout']}"
                return output
            else:
                return f"❌ Command failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Command execution error: {str(e)}"
    
    def _index_workspace_tool(self, input_str: str = "") -> str:
        try:
            if not self.rag_system:
                return "❌ RAG system not available"
            
            result = self.rag_system.index_workspace()
            return f"🗂️ Indexing result: {result['message']}"
            
        except Exception as e:
            return f"❌ Indexing error: {str(e)}"
    
    async def _reindex_background(self):
        """Re-index workspace in the background"""
        if self.rag_system:
            self.rag_system.index_workspace()
    
    async def process_query(self, query: str) -> str:
        """Process a user query"""
        try:
            if not self.agent_executor:
                return "❌ AI assistant not initialized"
            
            response = self.agent_executor.invoke({"input": query})
            return response.get("output", "No response generated")
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"


class AIFileAssistantApp(App):
    """Main Textual application"""
    
    CSS = """
    Screen {
        background: #1a1a1a;
    }
    
    Header {
        background: #ff6600;
        color: white;
        text-align: center;
        height: 3;
        content-align: center middle;
    }
    
    Footer {
        background: #333333;
        color: #ff6600;
        height: 3;
    }
    
    .main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    .sidebar {
        width: 30%;
        background: #2d2d2d;
        border-right: solid #ff6600;
    }
    
    .content {
        width: 70%;
        background: #1e1e1e;
    }
    
    .chat-area {
        height: 1fr;
        background: #1e1e1e;
        border: solid #ff6600;
        margin: 1;
    }
    
    .input-area {
        height: 3;
        background: #333333;
        border: solid #ff6600;
        margin: 1;
    }
    
    Input {
        background: #333333;
        color: #ff6600;
    }
    
    Input:focus {
        border: solid #ff9933;
    }
    
    Button {
        background: #ff6600;
        color: white;
    }
    
    Button:hover {
        background: #ff9933;
    }
    
    RichLog {
        background: #1e1e1e;
        color: #ffffff;
        border: solid #444444;
    }
    
    DirectoryTree {
        background: #2d2d2d;
        color: #ffffff;
    }
    
    Static {
        background: #1e1e1e;
        color: #ffffff;
    }
    
    #title {
        text-align: center;
        background: #ff6600;
        color: white;
        text-style: bold;
        height: 3;
    }
    
    #status {
        background: #333333;
        color: #ff6600;
        height: 1;
        text-align: center;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.ai_assistant = AIAssistantWidget()
        self.chat_log = None
        self.input_field = None
        self.status_bar = None
        self.directory_tree = None
        
    def compose(self) -> ComposeResult:
        """Create the application layout"""
        yield Header()
        
        # Title
        yield Static("🤖 AI FILE ASSISTANT\nPowered by Google Gemini & RAG", id="title")
        
        with Container(classes="main-container"):
            # Sidebar
            with Vertical(classes="sidebar"):
                yield Static("📁 Workspace", id="sidebar-title")
                yield DirectoryTree(str(Path(config.get_workspace_path()).resolve()), id="directory-tree")
                
                # Control buttons
                with Horizontal():
                    yield Button("🗂️ Index", id="index-btn", variant="primary")
                    yield Button("🔄 Refresh", id="refresh-btn", variant="default")
            
            # Main content area
            with Vertical(classes="content"):
                # Chat area
                yield RichLog(id="chat-log", classes="chat-area")
                
                # Input area
                with Horizontal(classes="input-area"):
                    yield Input(placeholder="Ask me anything about your files...", id="user-input")
                    yield Button("Send", id="send-btn", variant="primary")
        
        # Status bar
        yield Static("Initializing AI assistant...", id="status")
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize the application"""
        self.chat_log = self.query_one("#chat-log", RichLog)
        self.input_field = self.query_one("#user-input", Input)
        self.status_bar = self.query_one("#status", Static)
        self.directory_tree = self.query_one("#directory-tree", DirectoryTree)
        
        # Welcome message
        self.chat_log.write(Panel(
            Text("Welcome to AI File Assistant! 🤖\n\nThis advanced assistant can help you:\n• Create, read, and edit files\n• Search your workspace with RAG\n• Execute safe terminal commands\n• Understand your codebase context\n\nAll operations are sandboxed to your workspace.", 
                 style="bold orange1"),
            title="[bold orange1]Welcome[/bold orange1]",
            border_style="orange1"
        ))
        
        # Initialize AI
        self.status_bar.update("🔧 Setting up AI assistant...")
        success = await self.ai_assistant.setup_ai()
        
        if success:
            self.status_bar.update("✅ Ready! Type your question below.")
            
            # Initial indexing
            self.status_bar.update("🗂️ Indexing workspace...")
            self.ai_assistant.rag_system.index_workspace()
            self.status_bar.update("✅ Ready! Workspace indexed.")
            
        else:
            self.status_bar.update("❌ AI initialization failed. Check your API key.")
        
        # Focus input
        self.input_field.focus()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "send-btn":
            await self.send_message()
        elif event.button.id == "index-btn":
            await self.index_workspace()
        elif event.button.id == "refresh-btn":
            await self.refresh_directory()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "user-input":
            await self.send_message()
    
    async def send_message(self) -> None:
        """Send a message to the AI"""
        message = self.input_field.value.strip()
        if not message:
            return
        
        # Clear input
        self.input_field.value = ""
        
        # Display user message
        self.chat_log.write(Panel(
            Text(message, style="bold blue"),
            title="[bold blue]You[/bold blue]",
            border_style="blue"
        ))
        
        # Show thinking status
        self.status_bar.update("🤔 AI is thinking...")
        
        # Get AI response
        try:
            response = await self.ai_assistant.process_query(message)
            
            # Display AI response
            self.chat_log.write(Panel(
                Text(response, style="orange1"),
                title="[bold orange1]🤖 Assistant[/bold orange1]",
                border_style="orange1"
            ))
            
            self.status_bar.update("✅ Ready! Type your next question.")
            
        except Exception as e:
            self.chat_log.write(Panel(
                Text(f"Error: {str(e)}", style="red"),
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
            self.status_bar.update("❌ Error occurred. Try again.")
        
        # Focus input again
        self.input_field.focus()
    
    async def index_workspace(self) -> None:
        """Re-index the workspace"""
        self.status_bar.update("🗂️ Indexing workspace...")
        try:
            result = self.ai_assistant.rag_system.index_workspace()
            self.status_bar.update(f"✅ {result['message']}")
            
            self.chat_log.write(Panel(
                Text(f"Workspace re-indexed!\n{result['message']}", style="green"),
                title="[bold green]Index Complete[/bold green]",
                border_style="green"
            ))
        except Exception as e:
            self.status_bar.update("❌ Indexing failed")
            self.chat_log.write(Panel(
                Text(f"Indexing failed: {str(e)}", style="red"),
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
    
    async def refresh_directory(self) -> None:
        """Refresh the directory tree"""
        try:
            self.directory_tree.reload()
            self.status_bar.update("✅ Directory refreshed")
        except Exception as e:
            self.status_bar.update("❌ Failed to refresh directory")


def main():
    """Main entry point"""
    # Create workspace if it doesn't exist
    workspace_path = Path(config.get_workspace_path())
    workspace_path.mkdir(exist_ok=True)
    
    # Run the Textual app
    app = AIFileAssistantApp()
    app.run()


if __name__ == "__main__":
    main()
