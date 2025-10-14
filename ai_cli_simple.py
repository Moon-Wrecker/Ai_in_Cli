#!/usr/bin/env python3
"""
🤖 AI in CLI - Beautiful & Simple Terminal Interface

A stunning, simplified AI assistant with beautiful orange theme
and modern terminal interface.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Core imports
from dotenv import load_dotenv

# Textual imports
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Input, RichLog, Static, Button, Label
)
from textual import on, work

# Rich imports  
from rich.panel import Panel
from rich.align import Align
from rich.markdown import Markdown

# Local imports
from ai_assistant import AIFileAssistant

# Load environment variables
load_dotenv()


class AICliApp(App):
    """Beautiful AI in CLI Application - Simplified"""
    
    CSS = """
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
    
    #intro-panel {
        height: 1fr;
        background: #1a1a1a;
        border: solid #ff6b35;
        margin: 2;
        padding: 2;
    }
    
    #button-container {
        height: 5;
        background: #1a1a1a;
        align: center middle;
    }
    
    #chat-container {
        height: 1fr;
        background: #0a0a0a;
    }
    
    #chat-log {
        background: #0f0f0f;
        color: #ffffff;
        border: solid #333333;
        margin: 1;
        height: 1fr;
    }
    
    #input-container {
        height: 4;
        background: #1a1a1a;
        padding: 1;
    }
    
    #chat-input {
        background: #2a2a2a;
        color: #ffffff;
        border: solid #ff6b35;
        width: 80%;
    }
    
    #chat-input:focus {
        border: solid #f7931e;
        background: #3a3a3a;
    }
    
    Button {
        background: #ff6b35;
        color: white;
        margin: 0 2;
        min-width: 15;
        height: 3;
    }
    
    Button:hover {
        background: #f7931e;
        text-style: bold;
    }
    
    #send-btn {
        background: #f7931e;
        min-width: 8;
    }
    
    Static {
        background: transparent;
        color: #ffffff;
    }
    """
    
    TITLE = "🤖 AI in CLI"
    SUB_TITLE = "Intelligent Terminal Assistant"
    
    def __init__(self):
        super().__init__()
        self.ai_assistant: Optional[AIFileAssistant] = None
        self.show_intro = True
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        if self.show_intro:
            # Intro screen
            intro_text = """
# 🚀 Welcome to AI in CLI

**The most advanced AI-powered terminal assistant**

---

## ✨ Key Features

🤖 **Smart AI Assistant** - Powered by Google Gemini  
📁 **File Management** - Secure workspace operations  
🔍 **RAG Technology** - Workspace awareness & search  
⚡ **Smart Editing** - Line-level code modifications  
🛠️ **Safe Commands** - Sandboxed terminal operations  

---

## 🎯 Perfect For

• **Developers** managing codebases  
• **DevOps Engineers** automating tasks  
• **Data Scientists** organizing projects  
• **Power Users** seeking intelligent assistance  

---

## 💡 Try These Commands

*"List all Python files in my project"*  
*"Create a README with project overview"*  
*"Find all TODO comments"*  
*"Analyze my code structure"*  
*"Show system information"*

---

**Ready to experience intelligent file management?**
            """
            
            yield Static(Markdown(intro_text), id="intro-panel")
            with Container(id="button-container"):
                yield Button("🚀 Launch AI Assistant", id="start-btn")
                yield Button("❌ Exit", id="exit-btn")
        else:
            # Main chat interface
            yield Container(
                RichLog(id="chat-log", wrap=True, highlight=True, markup=True),
                id="chat-container"
            )
            with Horizontal(id="input-container"):
                yield Input(
                    placeholder="Ask me anything about your files, code, or system...", 
                    id="chat-input"
                )
                yield Button("Send", id="send-btn")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the application"""
        self.title = "🤖 AI in CLI"
        self.sub_title = "Intelligent Terminal Assistant"
    
    @on(Button.Pressed, "#start-btn")
    def start_assistant(self) -> None:
        """Start the AI assistant"""
        self.show_intro = False
        self.recompose()
        
        # Initialize AI in background and show loading
        chat_log = self.query_one("#chat-log", RichLog)
        loading_panel = Panel(
            Align.center(
                "[bold #ff6b35]🚀 Initializing AI Assistant...[/]\n\n"
                "[#f7931e]• Loading Google Gemini AI[/]\n"
                "[#f7931e]• Setting up RAG system[/]\n"  
                "[#f7931e]• Indexing workspace[/]\n"
                "[dim]Please wait a moment...[/]"
            ),
            border_style="#ff6b35",
            title="[bold #f7931e]Starting Up[/]"
        )
        chat_log.write(loading_panel)
        
        # Start initialization
        self.init_ai_assistant()
        
        # Focus on input
        self.query_one("#chat-input", Input).focus()
    
    @on(Button.Pressed, "#exit-btn")
    def exit_app(self) -> None:
        """Exit the application"""
        self.exit()
    
    @work(exclusive=True)
    async def init_ai_assistant(self) -> None:
        """Initialize AI assistant in background"""
        try:
            self.ai_assistant = AIFileAssistant()
            chat_log = self.query_one("#chat-log", RichLog)
            
            success_panel = Panel(
                Align.center(
                    "[bold green]✅ AI Assistant Ready![/]\n\n"
                    f"[dim]📁 Workspace: {self.ai_assistant.workspace_dir}[/]\n"
                    "[dim]🔍 RAG System: Active[/]\n"
                    "[dim]⚡ Smart Editor: Ready[/]\n\n"
                    "[green]🎯 Ready for your commands![/]"
                ),
                border_style="green",
                title="[bold green]System Ready[/]"
            )
            chat_log.write(success_panel)
            
        except Exception as e:
            chat_log = self.query_one("#chat-log", RichLog)
            error_panel = Panel(
                f"[bold red]❌ Initialization failed:[/]\n\n{str(e)}\n\n"
                "[yellow]You can still try commands, but functionality may be limited.[/]",
                border_style="red",
                title="[bold red]Error[/]"
            )
            chat_log.write(error_panel)
    
    @on(Button.Pressed, "#send-btn")
    @on(Input.Submitted, "#chat-input")
    def send_message(self) -> None:
        """Send message to AI"""
        if self.show_intro:
            return
            
        chat_input = self.query_one("#chat-input", Input)
        message = chat_input.value.strip()
        
        if not message:
            return
        
        chat_input.value = ""
        chat_log = self.query_one("#chat-log", RichLog)
        
        # Show user message
        user_panel = Panel(
            message,
            border_style="#4a90e2", 
            title="[bold #4a90e2]👤 You[/]",
            title_align="left"
        )
        chat_log.write(user_panel)
        
        # Show thinking
        thinking_panel = Panel(
            "[bold #ff6b35]🤔 Processing your request...[/]",
            border_style="#ff6b35",
            title="[bold #ff6b35]AI Thinking[/]"
        )
        chat_log.write(thinking_panel)
        
        # Process in background
        self.process_command(message)
    
    @work(exclusive=True)
    async def process_command(self, message: str) -> None:
        """Process AI command"""
        try:
            chat_log = self.query_one("#chat-log", RichLog)
            
            if not self.ai_assistant:
                error_panel = Panel(
                    "[bold red]AI Assistant not ready yet. Please wait for initialization.[/]",
                    border_style="red",
                    title="[bold red]Not Ready[/]"
                )
                chat_log.write(error_panel)
                return
            
            # Get AI response
            response = self.ai_assistant.process_command(message)
            
            # Show response
            ai_panel = Panel(
                response,
                border_style="#ff6b35",
                title="[bold #ff6b35]🤖 AI Assistant[/]",
                title_align="left"
            )
            chat_log.write(ai_panel)
            
        except Exception as e:
            chat_log = self.query_one("#chat-log", RichLog)
            error_panel = Panel(
                f"[bold red]Error:[/] {str(e)}",
                border_style="red",
                title="[bold red]Command Failed[/]"
            )
            chat_log.write(error_panel)


def run():
    """Run the beautiful AI CLI"""
    app = AICliApp()
    app.run()


if __name__ == "__main__":
    run()
