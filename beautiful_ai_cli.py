#!/usr/bin/env python3
"""
🤖 AI in CLI - Beautiful & Simple

A beautiful, working AI assistant with Textual interface.
"""

import os
from pathlib import Path
from typing import Optional

# Textual imports
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static, Button
from textual import on

# Rich imports
from rich.panel import Panel
from rich.align import Align

# Load environment
from dotenv import load_dotenv
load_dotenv()


class IntroScreen(Static):
    """Intro screen widget"""
    
    def render(self) -> str:
        return """
[bold #ff6b35]🚀 Welcome to AI in CLI[/]

[#f7931e]The most advanced AI-powered terminal assistant[/]

─────────────────────────────────

[bold]✨ Key Features[/]
• 🤖 Smart AI Assistant (Google Gemini)
• 📁 Secure File Management  
• 🔍 RAG-Powered Workspace Search
• ⚡ Intelligent Code Editing
• 🛠️ Safe Terminal Operations

[bold]🎯 Perfect For[/]
• Developers managing codebases
• DevOps Engineers automating tasks
• Data Scientists organizing projects
• Anyone seeking intelligent assistance

[bold]💡 Example Commands[/]
• "List all Python files"
• "Create a project README"
• "Find TODO comments"
• "Analyze code structure"
• "Show system info"

─────────────────────────────────

[bold #ff6b35]Ready to experience the future of terminal interaction?[/]
        """


class AICliApp(App):
    """Beautiful AI in CLI Application"""
    
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
    
    IntroScreen {
        background: #1a1a1a;
        color: #ffffff;
        border: solid #ff6b35;
        margin: 2;
        padding: 3;
        text-align: center;
    }
    
    .buttons {
        height: 5;
        background: #1a1a1a;
        align: center middle;
        margin: 1;
    }
    
    Button {
        background: #ff6b35;
        color: white;
        margin: 0 2;
        min-width: 18;
        height: 3;
    }
    
    Button:hover {
        background: #f7931e;
        text-style: bold;
    }
    
    #chat-area {
        background: #0a0a0a;
        height: 1fr;
    }
    
    #chat-log {
        background: #0f0f0f;
        color: #ffffff;
        border: solid #333333;
        margin: 1;
        height: 1fr;
    }
    
    .input-row {
        height: 4;
        background: #1a1a1a;
        padding: 1;
    }
    
    #message-input {
        background: #2a2a2a;
        color: #ffffff;
        border: solid #ff6b35;
        width: 85%;
    }
    
    #message-input:focus {
        border: solid #f7931e;
        background: #3a3a3a;
    }
    
    #send-btn {
        background: #f7931e;
        min-width: 10;
    }
    """
    
    TITLE = "🤖 AI in CLI"
    SUB_TITLE = "Intelligent Terminal Assistant"
    
    def __init__(self):
        super().__init__()
        self.ai_assistant = None
        self.in_chat_mode = False
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        if not self.in_chat_mode:
            # Show intro screen
            yield IntroScreen()
            with Container(classes="buttons"):
                yield Button("🚀 Launch AI Assistant", id="launch-btn")
                yield Button("❌ Exit", id="exit-btn")
        else:
            # Show chat interface
            with Container(id="chat-area"):
                yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
            with Horizontal(classes="input-row"):
                yield Input(placeholder="Type your command here...", id="message-input")
                yield Button("Send", id="send-btn")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """App startup"""
        self.title = "🤖 AI in CLI - Intelligent Terminal Assistant"
    
    @on(Button.Pressed, "#launch-btn")
    def launch_assistant(self) -> None:
        """Launch the AI assistant"""
        self.in_chat_mode = True
        self.recompose()
        
        # Show welcome message
        chat_log = self.query_one("#chat-log")
        
        welcome = Panel(
            Align.center(
                "[bold #ff6b35]🤖 AI in CLI Assistant[/]\n\n"
                "[#f7931e]Initializing AI systems...[/]\n"
                "[dim]Please wait while I set up your workspace[/]"
            ),
            border_style="#ff6b35",
            title="[bold #f7931e]Starting Up[/]"
        )
        chat_log.write(welcome)
        
        # Initialize AI assistant
        self.call_later(self._init_assistant)
        
        # Focus input
        self.query_one("#message-input").focus()
    
    def _init_assistant(self) -> None:
        """Initialize AI assistant"""
        try:
            from ai_assistant import AIFileAssistant
            self.ai_assistant = AIFileAssistant()
            
            chat_log = self.query_one("#chat-log")
            success = Panel(
                Align.center(
                    "[bold green]✅ AI Assistant Ready![/]\n\n"
                    f"[dim]📁 Workspace: {self.ai_assistant.workspace_dir}[/]\n"
                    "[dim]🔍 RAG System: Active[/]\n"
                    "[dim]⚡ Smart Editor: Ready[/]\n\n"
                    "[green]🎯 Ready for your commands![/]"
                ),
                border_style="green",
                title="[bold green]System Online[/]"
            )
            chat_log.write(success)
            
        except Exception as e:
            chat_log = self.query_one("#chat-log")
            error = Panel(
                f"[bold red]❌ Setup failed:[/]\n\n{str(e)}\n\n"
                "[yellow]Limited functionality available[/]",
                border_style="red", 
                title="[bold red]Error[/]"
            )
            chat_log.write(error)
    
    @on(Button.Pressed, "#exit-btn")
    def exit_application(self) -> None:
        """Exit the app"""
        self.exit()
    
    @on(Button.Pressed, "#send-btn")
    @on(Input.Submitted, "#message-input")
    def send_message(self) -> None:
        """Send message to AI"""
        if not self.in_chat_mode:
            return
        
        input_widget = self.query_one("#message-input")
        message = input_widget.value.strip()
        
        if not message:
            return
        
        input_widget.value = ""
        chat_log = self.query_one("#chat-log")
        
        # Show user message
        user_msg = Panel(
            message,
            border_style="#4a90e2",
            title="[bold #4a90e2]👤 You[/]"
        )
        chat_log.write(user_msg)
        
        # Show processing
        processing = Panel(
            "[bold #ff6b35]🤔 Processing your request...[/]",
            border_style="#ff6b35",
            title="[bold #ff6b35]AI Thinking[/]"
        )
        chat_log.write(processing)
        
        # Process command
        self.call_later(self._process_message, message)
    
    def _process_message(self, message: str) -> None:
        """Process AI message"""
        chat_log = self.query_one("#chat-log")
        
        try:
            if not self.ai_assistant:
                response = "AI Assistant not initialized. Please restart the application."
            else:
                response = self.ai_assistant.process_command(message)
            
            # Show AI response
            ai_msg = Panel(
                response,
                border_style="#ff6b35",
                title="[bold #ff6b35]🤖 AI Assistant[/]"
            )
            chat_log.write(ai_msg)
            
        except Exception as e:
            error_msg = Panel(
                f"[bold red]Error:[/] {str(e)}",
                border_style="red",
                title="[bold red]Command Failed[/]"
            )
            chat_log.write(error_msg)


def main():
    """Launch the beautiful AI CLI"""
    app = AICliApp()
    app.run()


if __name__ == "__main__":
    main()
