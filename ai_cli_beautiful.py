#!/usr/bin/env python3
"""
🤖 AI in CLI - Beautiful Terminal AI Assistant

A stunning, modern terminal application with AI-powered file management,
workspace awareness, and intelligent conversation capabilities.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

# Core imports
from dotenv import load_dotenv

# Textual imports
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Input, RichLog, Static, Button, 
    Label, LoadingIndicator, Markdown
)
from textual.reactive import reactive
from textual import on, work
from textual.message import Message

# Rich imports
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich.table import Table
from rich.markdown import Markdown as RichMarkdown

# Local imports
from ai_assistant import AIFileAssistant

# Load environment variables
load_dotenv()


class IntroScreen(Static):
    """Beautiful introduction screen with capabilities"""
    
    def compose(self) -> ComposeResult:
        intro_content = """
# 🚀 Welcome to AI in CLI

**The most advanced AI-powered terminal assistant for developers and power users**

---

## ✨ Core Capabilities

### 🤖 **Intelligent AI Assistant**
- Powered by Google Gemini for natural language understanding
- Context-aware conversations with memory
- Smart command interpretation and execution

### 📁 **Advanced File Management**
- Secure workspace sandboxing
- Smart file operations with line-level precision
- Intelligent code analysis and structure understanding

### 🔍 **RAG-Powered Workspace Awareness**
- Semantic search across your entire workspace
- Intelligent code understanding and context
- Project-aware suggestions and recommendations

### ⚡ **Smart Code Editing**
- Line-by-line file editing with surgical precision
- Multi-language code analysis and understanding
- Intelligent refactoring and code suggestions

### 🛠️ **Terminal Operations**
- Safe command execution with sandboxing
- System information and process management
- Intelligent command safety validation

---

## 🎯 **Perfect For**
- **Developers** managing complex codebases
- **DevOps Engineers** automating file operations  
- **Data Scientists** organizing research projects
- **Anyone** who wants an intelligent terminal companion

---

## 💡 **Example Commands**
- *"Create a Python script for data analysis"*
- *"Find all TODO comments in my code"*
- *"Analyze the structure of my project"*
- *"Refactor this function to be more efficient"*
- *"Show me system performance metrics"*

---

**Ready to experience the future of terminal interaction?**
"""
        
        yield Markdown(intro_content, id="intro-content")


class ChatInterface(Container):
    """Main chat interface with AI assistant"""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🤖 AI Assistant", classes="section-header")
            yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
            yield Horizontal(
                Input(placeholder="Ask me anything about your files, code, or system...", id="chat-input"),
                Button("Send", variant="primary", id="send-btn"),
                classes="input-area"
            )


class WorkspacePanel(Container):
    """Workspace information and file browser"""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📁 Workspace Explorer", classes="section-header")
            yield Static("", id="workspace-info")
            yield Button("🔄 Refresh Workspace", id="refresh-workspace")


class AICliApp(App):
    """Beautiful AI in CLI Application"""
    
    CSS = """
    /* Orange Theme Styling */
    Screen {
        background: #0a0a0a;
    }
    
    Header {
        dock: top;
        height: 3;
        background: #ff6b35;
        color: white;
        text-align: center;
    }
    
    Footer {
        dock: bottom;
        height: 3;
        background: #1a1a1a;
        color: #ff6b35;
        text-align: center;
    }
    
    .section-header {
        background: #ff6b35;
        color: white;
        text-align: center;
        height: 3;
        content-align: center middle;
        text-style: bold;
    }
    
    #intro-content {
        background: #1a1a1a;
        color: #ffffff;
        border: solid #ff6b35;
        margin: 1;
        padding: 2;
    }
    
    #intro-buttons {
        height: 5;
        dock: bottom;
        background: #1a1a1a;
        align-horizontal: center;
        padding: 1;
    }
    
    #chat-log {
        background: #0f0f0f;
        border: solid #333333;
        color: #ffffff;
        height: 1fr;
    }
    
    .input-area {
        height: 3;
        background: #1a1a1a;
        padding: 1;
    }
    
    #chat-input {
        background: #2a2a2a;
        color: #ffffff;
        border: solid #ff6b35;
    }
    
    #chat-input:focus {
        border: solid #f7931e;
        background: #3a3a3a;
    }
    
    Button {
        background: #ff6b35;
        color: white;
        border: none;
        min-width: 12;
        height: 3;
        margin: 0 1;
    }
    
    Button:hover {
        background: #f7931e;
        text-style: bold;
    }
    
    Button.-primary {
        background: #f7931e;
    }
    
    #workspace-info {
        background: #1a1a1a;
        color: #cccccc;
        border: solid #333333;
        padding: 1;
        height: 10;
    }
    
    #left-panel {
        width: 25%;
        background: #111111;
    }
    
    #main-panel {
        width: 75%;
        background: #0a0a0a;
    }
    """
    
    TITLE = "🤖 AI in CLI - Intelligent Terminal Assistant"
    SUB_TITLE = "Powered by Google Gemini & RAG Technology"
    
    show_intro = reactive(True)
    ai_assistant: Optional[AIFileAssistant] = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        if self.show_intro:
            yield IntroScreen(id="intro-screen")
            yield Container(
                Button("🚀 Start AI Assistant", variant="primary", id="start-btn"),
                Button("ℹ️ Show Capabilities", id="info-btn"),
                Button("❌ Exit", id="exit-btn"),
                id="intro-buttons"
            )
        else:
            with Horizontal():
                with Vertical(id="left-panel"):
                    yield WorkspacePanel(id="workspace-panel")
                with Vertical(id="main-panel"):
                    yield ChatInterface(id="chat-interface")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the application"""
        self.title = "🤖 AI in CLI"
        self.sub_title = "Intelligent Terminal Assistant"
        
        # Update footer with helpful info
        footer_text = "Press [bold #ff6b35]Ctrl+C[/] to exit • [bold #f7931e]AI-powered file management[/] • [bold #ff6b35]RAG-enabled workspace awareness[/]"
        if hasattr(self.query_one(Footer), 'update'):
            self.query_one(Footer).update(footer_text)
    
    @on(Button.Pressed, "#start-btn")
    def start_assistant(self) -> None:
        """Start the AI assistant"""
        self.show_intro = False
        self.recompose()
        
        # Initialize AI assistant in background
        self.init_ai_assistant()
        
        # Show welcome message
        chat_log = self.query_one("#chat-log", RichLog)
        welcome_panel = Panel(
            Align.center(
                "[bold #ff6b35]🤖 AI Assistant Starting...[/]\n\n"
                "[#f7931e]Initializing RAG system and workspace awareness...[/]\n"
                "[dim]Please wait a moment...[/]"
            ),
            border_style="#ff6b35",
            title="[bold #f7931e]Starting Up[/]"
        )
        chat_log.write(welcome_panel)
        
        # Update workspace info
        self.update_workspace_info()
        
        # Focus on input
        self.query_one("#chat-input", Input).focus()
    
    @on(Button.Pressed, "#info-btn")
    def show_info(self) -> None:
        """Show application information"""
        pass  # Info already shown in intro
    
    @on(Button.Pressed, "#exit-btn")
    def exit_app(self) -> None:
        """Exit the application"""
        self.exit()
    
    @work(exclusive=True)
    async def init_ai_assistant(self) -> None:
        """Initialize the AI assistant in the background"""
        try:
            # Import and initialize in background
            self.ai_assistant = AIFileAssistant()
            
            # Show success message in chat
            chat_log = self.query_one("#chat-log", RichLog)
            
            success_panel = Panel(
                "[bold green]✅ AI Assistant Ready![/]\n"
                f"[dim]Workspace: {self.ai_assistant.workspace_dir}[/]\n"
                "[dim]RAG System: Initialized[/]\n"
                "[dim]Smart Editor: Ready[/]\n"
                "[green]Type your first command below![/]",
                border_style="green",
                title="[bold green]System Ready[/]"
            )
            chat_log.write(success_panel)
            
        except Exception as e:
            chat_log = self.query_one("#chat-log", RichLog)
            error_panel = Panel(
                f"[bold red]❌ Initialization Error:[/]\n{str(e)}\n"
                "[dim]You can still try to send commands...[/]",
                border_style="red",
                title="[bold red]Error[/]"
            )
            chat_log.write(error_panel)
    
    @on(Button.Pressed, "#send-btn")
    @on(Input.Submitted, "#chat-input")
    async def send_message(self) -> None:
        """Send message to AI assistant"""
        chat_input = self.query_one("#chat-input", Input)
        user_message = chat_input.value.strip()
        
        if not user_message:
            return
        
        chat_input.value = ""
        chat_log = self.query_one("#chat-log", RichLog)
        
        # Show user message
        user_panel = Panel(
            user_message,
            border_style="#4a90e2",
            title="[bold #4a90e2]You[/]",
            title_align="left"
        )
        chat_log.write(user_panel)
        
        if not self.ai_assistant:
            error_panel = Panel(
                "[bold red]AI Assistant not initialized yet. Please wait...[/]",
                border_style="red"
            )
            chat_log.write(error_panel)
            return
        
        # Show thinking indicator
        with chat_log.batch():
            thinking_panel = Panel(
                "[bold #ff6b35]🤔 AI is thinking...[/]",
                border_style="#ff6b35",
                title="[bold #ff6b35]Processing[/]"
            )
            chat_log.write(thinking_panel)
        
        # Process command
        await self.process_ai_command(user_message, chat_log)
    
    @work(exclusive=True)
    async def process_ai_command(self, message: str, chat_log: RichLog) -> None:
        """Process AI command in the background"""
        try:
            if not self.ai_assistant:
                error_panel = Panel(
                    "[bold red]AI Assistant not initialized[/]",
                    border_style="red"
                )
                chat_log.write(error_panel)
                return
                
            # Get AI response
            response = self.ai_assistant.process_command(message)
            
            # Show AI response
            ai_panel = Panel(
                response,
                border_style="#ff6b35",
                title="[bold #ff6b35]🤖 AI Assistant[/]",
                title_align="left"
            )
            chat_log.write(ai_panel)
            
            # Update workspace info after operations
            self.update_workspace_info()
            
        except Exception as e:
            error_panel = Panel(
                f"[bold red]Error processing command:[/]\n{str(e)}",
                border_style="red",
                title="[bold red]Error[/]"
            )
            chat_log.write(error_panel)
    
    @on(Button.Pressed, "#refresh-workspace")
    def refresh_workspace(self) -> None:
        """Refresh workspace information"""
        self.update_workspace_info()
    
    def update_workspace_info(self) -> None:
        """Update workspace information panel"""
        try:
            workspace_info = self.query_one("#workspace-info", Static)
            
            if self.ai_assistant:
                workspace_path = self.ai_assistant.workspace_dir
                
                # Get workspace structure
                structure = self.ai_assistant.fs_manager.folder_structure(".")
                
                # Create info text
                info_text = f"[bold #ff6b35]📁 Current Workspace[/]\n"
                info_text += f"[dim]{workspace_path}[/]\n\n"
                info_text += f"[bold #f7931e]📊 Workspace Contents[/]\n"
                
                # Count files and folders
                total_files = 0
                total_folders = 0
                
                def count_items(struct):
                    nonlocal total_files, total_folders
                    for key, value in struct.items():
                        if key == '_files':
                            total_files += len(value)
                        else:
                            total_folders += 1
                            if isinstance(value, dict):
                                count_items(value)
                
                count_items(structure)
                
                info_text += f"📄 Files: [bold]{total_files}[/]\n"
                info_text += f"📁 Folders: [bold]{total_folders}[/]\n\n"
                info_text += f"[dim]Last updated: {datetime.now().strftime('%H:%M:%S')}[/]"
                
                workspace_info.update(info_text)
            else:
                workspace_info.update("[dim]AI Assistant not initialized[/]")
                
        except Exception as e:
            workspace_info = self.query_one("#workspace-info", Static)
            workspace_info.update(f"[red]Error: {str(e)}[/]")


def run_beautiful_ai_cli():
    """Run the beautiful AI CLI application"""
    app = AICliApp()
    app.run()


if __name__ == "__main__":
    run_beautiful_ai_cli()
