#!/usr/bin/env python3
"""
AI CLI - Intelligent Terminal Assistant
Main entry point with beautiful Rich UI
"""

import os
import sys
import signal
import argparse
from pathlib import Path
from typing import Optional

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from rich import box

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize console
console = Console()


def get_config():
    """Get configuration - import here to avoid circular imports"""
    from config import get_settings, get_sandbox_path, APP_NAME, APP_VERSION
    return get_settings(), get_sandbox_path(), APP_NAME, APP_VERSION


def print_banner():
    """Print the application banner"""
    _, _, APP_NAME, APP_VERSION = get_config()
    
    banner_text = Text()
    banner_text.append("╔════════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║", style="bold cyan")
    banner_text.append("               🤖 AI CLI Assistant                  ", style="bold white")
    banner_text.append("║\n", style="bold cyan")
    banner_text.append("║", style="bold cyan")
    banner_text.append(f"          Claude Code-level AI Assistant            ", style="dim white")
    banner_text.append("║\n", style="bold cyan")
    banner_text.append("║", style="bold cyan")
    banner_text.append(f"                    v{APP_VERSION}                          ", style="dim cyan")
    banner_text.append("║\n", style="bold cyan")
    banner_text.append("╚════════════════════════════════════════════════════╝", style="bold cyan")
    
    console.print(banner_text)
    console.print()


def print_workspace_info():
    """Print workspace information"""
    _, sandbox, _, _ = get_config()
    console.print(f"📁 [bold]Workspace:[/bold] [cyan]{sandbox}[/cyan]")
    console.print()


def print_help_info():
    """Print quick help"""
    help_text = """[dim]Quick Commands:
  • Type your request in natural language
  • [cyan]/help[/cyan]     - Show all commands
  • [cyan]/clear[/cyan]    - Clear conversation
  • [cyan]/index[/cyan]    - Index workspace for search
  • [cyan]/stats[/cyan]    - Show statistics
  • [cyan]/tools[/cyan]    - List available tools
  • [cyan]/exit[/cyan]     - Exit the assistant
[/dim]"""
    console.print(help_text)


def handle_special_command(command: str, agent) -> bool:
    """
    Handle special commands starting with /
    
    Returns:
        True if command was handled, False otherwise
    """
    command = command.strip().lower()
    
    if command == "/help":
        help_table = Table(title="Available Commands", box=box.ROUNDED)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description")
        
        help_table.add_row("/help", "Show this help message")
        help_table.add_row("/clear", "Clear conversation history")
        help_table.add_row("/index", "Full workspace indexing")
        help_table.add_row("/index-status", "Check index status & pending changes")
        help_table.add_row("/stats", "Show agent statistics")
        help_table.add_row("/tools", "List available tools")
        help_table.add_row("/workspace", "Show workspace info")
        help_table.add_row("/exit or /quit", "Exit the assistant")
        
        console.print(help_table)
        return True
    
    elif command == "/clear":
        agent.clear_memory()
        console.print("[green]✓ Conversation cleared[/green]")
        return True
    
    elif command == "/index":
        console.print("[yellow]🔄 Indexing workspace... (this may take a moment)[/yellow]")
        
        try:
            from tools.search_tools import SearchTools
            tools = SearchTools()
            result = tools.index_workspace(force=True)
            
            if "error" in result:
                console.print(f"[red]Error: {result['error']}[/red]")
            else:
                console.print("[green]✓ Workspace indexed successfully![/green]")
                
                ast = result.get("ast_indexing", {})
                console.print(f"  📄 Files processed: {ast.get('files_processed', 0)}")
                console.print(f"  🔣 Symbols found: {ast.get('symbols_found', 0)}")
                
                semantic = result.get("semantic_indexing", {})
                console.print(f"  📦 Chunks indexed: {semantic.get('chunks_indexed', 0)}")
        except Exception as e:
            console.print(f"[red]Error during indexing: {e}[/red]")
        
        return True
    
    elif command == "/index-status":
        try:
            from indexing.index_manager import get_index_manager
            manager = get_index_manager()
            
            stats = manager.get_stats()
            changes = manager.get_changed_files()
            
            status_table = Table(title="Index Status", box=box.ROUNDED)
            status_table.add_column("Metric", style="cyan")
            status_table.add_column("Value", style="green")
            
            status_table.add_row("Indexed Files", str(stats["indexed_files"]))
            status_table.add_row("Last Full Index", stats["last_full_index"] or "Never")
            status_table.add_row("New Files", str(len(changes["new"])))
            status_table.add_row("Modified Files", str(len(changes["modified"])))
            status_table.add_row("Deleted Files", str(len(changes["deleted"])))
            status_table.add_row("Needs Re-index", "Yes" if manager.needs_indexing() else "No")
            
            console.print(status_table)
            
            if changes["new"]:
                console.print(f"\n[dim]New files: {', '.join(p.name for p in changes['new'][:5])}{'...' if len(changes['new']) > 5 else ''}[/dim]")
            if changes["modified"]:
                console.print(f"[dim]Modified: {', '.join(p.name for p in changes['modified'][:5])}{'...' if len(changes['modified']) > 5 else ''}[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        
        return True
    
    elif command == "/stats":
        stats = agent.get_stats()
        
        stats_table = Table(title="Agent Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Model", stats["model"])
        stats_table.add_row("Temperature", str(stats["temperature"]))
        stats_table.add_row("Total Tokens Used", str(stats["total_tokens_used"]))
        stats_table.add_row("Tool Calls Made", str(stats["total_tool_calls"]))
        stats_table.add_row("Available Tools", str(stats["available_tools"]))
        stats_table.add_row("Messages in Memory", str(stats["memory"]["total_messages"]))
        
        console.print(stats_table)
        return True
    
    elif command == "/tools":
        from core.function_registry import get_registry
        registry = get_registry()
        
        tools_table = Table(title="Available Tools (25 total)", box=box.ROUNDED)
        tools_table.add_column("Tool", style="cyan", width=25)
        tools_table.add_column("Description")
        
        for tool_name in sorted(registry.list_tools()):
            tool = registry.get_tool(tool_name)
            if tool:
                desc = tool.description[:55] + "..." if len(tool.description) > 55 else tool.description
                tools_table.add_row(tool_name, desc)
        
        console.print(tools_table)
        return True
    
    elif command == "/workspace":
        print_workspace_info()
        
        _, sandbox, _, _ = get_config()
        
        # Show contents
        console.print("[bold]Contents:[/bold]")
        try:
            items = list(sandbox.iterdir())
            folders = [i for i in items if i.is_dir() and not i.name.startswith(".")]
            files = [i for i in items if i.is_file() and not i.name.startswith(".")]
            
            for folder in sorted(folders)[:15]:
                console.print(f"  📁 {folder.name}/")
            
            for file in sorted(files)[:15]:
                console.print(f"  📄 {file.name}")
            
            if len(folders) + len(files) > 30:
                console.print(f"  [dim]... and {len(folders) + len(files) - 30} more[/dim]")
            elif len(folders) + len(files) == 0:
                console.print("  [dim](empty)[/dim]")
                
        except Exception as e:
            console.print(f"  [red]Error listing: {e}[/red]")
        
        return True
    
    elif command in ["/exit", "/quit", "/q"]:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
        sys.exit(0)
    
    return False


def run_incremental_index_on_startup():
    """Run incremental indexing on startup if needed"""
    try:
        from indexing.index_manager import get_index_manager
        manager = get_index_manager()
        
        if manager.needs_indexing():
            console.print("[dim]🔄 Checking for file changes...[/dim]")
            result = manager.incremental_index(verbose=False)
            
            total_changes = (
                result["new_files"] + 
                result["modified_files"] + 
                result["deleted_files"]
            )
            
            if total_changes > 0:
                console.print(f"[dim]   ✓ Indexed {result['new_files']} new, "
                            f"{result['modified_files']} modified, "
                            f"{result['deleted_files']} deleted files[/dim]")
    except Exception:
        pass  # Don't fail startup if indexing fails


def run_interactive():
    """Run the interactive chat loop"""
    # Check for API key
    settings, sandbox, _, _ = get_config()
    
    if not settings.openai_api_key:
        console.print(Panel(
            "[red bold]Error: OPENAI_API_KEY not found![/red bold]\n\n"
            "Please set your OpenAI API key:\n"
            "1. Create a [cyan].env[/cyan] file in the [cyan]ai_cli[/cyan] directory\n"
            "2. Add: [green]OPENAI_API_KEY=your_key_here[/green]\n\n"
            "Get your API key from: [blue]https://platform.openai.com/api-keys[/blue]",
            title="Configuration Error",
            border_style="red"
        ))
        sys.exit(1)
    
    # Initialize agent
    try:
        from core.agent import AIAgent
        agent = AIAgent()
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        sys.exit(1)
    
    # Print banner and info
    print_banner()
    print_workspace_info()
    
    # Run incremental indexing on startup
    run_incremental_index_on_startup()
    
    print_help_info()
    
    # Main loop
    while True:
        try:
            # Get user input
            console.print()
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input.startswith("/"):
                if handle_special_command(user_input, agent):
                    continue
            
            # Process message
            console.print()
            console.print("[bold green]Assistant[/bold green]")
            
            full_response = ""
            
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                live.update(Text("🤔 Thinking...", style="yellow"))
                
                try:
                    # Get streaming response
                    gen = agent.chat_stream(user_input)
                    
                    for chunk in gen:
                        if chunk:
                            full_response += chunk
                            # Update display
                            live.update(Markdown(full_response))
                    
                except GeneratorExit:
                    pass
                except Exception as e:
                    live.update(Text(f"Error: {e}", style="red"))
                    console.print(f"[dim]Debug: {type(e).__name__}[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Press Ctrl+C again or type /exit to quit[/dim]")
            try:
                continue
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
        except EOFError:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break


def run_single_message(message: str, stream: bool = True):
    """Run a single message and exit"""
    settings, _, _, _ = get_config()
    
    if not settings.openai_api_key:
        console.print("[red]Error: OPENAI_API_KEY not found![/red]")
        sys.exit(1)
    
    from core.agent import AIAgent
    agent = AIAgent()
    
    if stream:
        console.print("[bold green]Assistant[/bold green]")
        full_response = ""
        
        with Live(console=console, refresh_per_second=10) as live:
            for chunk in agent.chat_stream(message):
                if chunk:
                    full_response += chunk
                    live.update(Markdown(full_response))
    else:
        response = agent.chat(message)
        console.print(Markdown(response.content))


def run_index(force: bool = False):
    """Index the workspace"""
    console.print("[yellow]🔄 Indexing workspace...[/yellow]")
    
    try:
        from tools.search_tools import SearchTools
        tools = SearchTools()
        result = tools.index_workspace(force=force)
        
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            sys.exit(1)
        
        console.print("[green]✓ Workspace indexed successfully![/green]")
        
        ast = result.get("ast_indexing", {})
        console.print(f"  Files processed: {ast.get('files_processed', 0)}")
        console.print(f"  Symbols found: {ast.get('symbols_found', 0)}")
        
        semantic = result.get("semantic_indexing", {})
        console.print(f"  Chunks indexed: {semantic.get('chunks_indexed', 0)}")
        
        graph = result.get("dependency_graph", {})
        console.print(f"  Graph nodes: {graph.get('total_nodes', 0)}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_search(query: str, limit: int = 10):
    """Search the codebase"""
    try:
        from tools.search_tools import SearchTools
        tools = SearchTools()
        
        result = tools.search(query, n_results=limit)
        
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            sys.exit(1)
        
        console.print(f"\n[bold]Found {result['result_count']} results for '{query}'[/bold]\n")
        
        for i, r in enumerate(result["results"], 1):
            console.print(f"[cyan]{i}. {r['file']}:{r['start_line']}-{r['end_line']}[/cyan]")
            console.print(f"   Score: {r['score']:.3f}")
            
            # Show preview
            preview = r["content"][:150].replace("\n", " ")
            if len(r["content"]) > 150:
                preview += "..."
            console.print(f"   [dim]{preview}[/dim]\n")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_tools_list():
    """List available tools"""
    from core.function_registry import get_registry
    registry = get_registry()
    
    tools_table = Table(title="Available Tools", box=box.ROUNDED)
    tools_table.add_column("Tool", style="cyan")
    tools_table.add_column("Description")
    
    for tool_name in sorted(registry.list_tools()):
        tool = registry.get_tool(tool_name)
        if tool:
            tools_table.add_row(tool_name, tool.description)
    
    console.print(tools_table)


def run_version():
    """Show version information"""
    _, sandbox, APP_NAME, APP_VERSION = get_config()
    console.print(f"[bold]{APP_NAME}[/bold] v{APP_VERSION}")
    console.print(f"Sandbox: {sandbox}")


def main():
    """Main entry point"""
    # Handle signals
    def signal_handler(sig, frame):
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="AI CLI - Intelligent Terminal Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start interactive mode
  python main.py chat "Hello"       # Send a single message
  python main.py index              # Index the workspace
  python main.py search "function"  # Search the codebase
  python main.py tools              # List available tools
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Send a message to the assistant")
    chat_parser.add_argument("message", nargs="?", help="Message to send (interactive if omitted)")
    chat_parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index the workspace for search")
    index_parser.add_argument("-f", "--force", action="store_true", help="Force re-index")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the codebase")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    
    # Tools command
    subparsers.add_parser("tools", help="List available tools")
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    # Execute command
    if args.command is None:
        # No command = interactive mode
        run_interactive()
    elif args.command == "chat":
        if args.message:
            run_single_message(args.message, stream=not args.no_stream)
        else:
            run_interactive()
    elif args.command == "index":
        run_index(force=args.force)
    elif args.command == "search":
        run_search(args.query, limit=args.limit)
    elif args.command == "tools":
        run_tools_list()
    elif args.command == "version":
        run_version()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
