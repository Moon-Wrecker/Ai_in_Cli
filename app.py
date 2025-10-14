#!/usr/bin/env python3
"""
🤖 AI in CLI - Beautiful Terminal AI Assistant

Main entry point for the beautiful AI-powered file assistant.
"""

import os
import sys
import warnings
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """Main entry point - Launch the beautiful AI assistant"""
    console = Console()
    
    # Display beautiful banner
    banner = Text("AI in CLI", style="bold orange1", justify="center")
    subtitle = Text("Intelligent Terminal Assistant", style="orange3", justify="center")
    
    banner_panel = Panel(
        Align.center(banner + "\n" + subtitle),
        border_style="orange1",
        padding=(1, 3),
        title="🤖",
        title_align="center"
    )
    
    console.print()
    console.print(banner_panel)
    
    # Show workspace info
    workspace_path = Path("workspace").resolve()
    console.print(f"📁 [orange3]{workspace_path}[/orange3]")
    console.print()
    
    # Launch console interface directly (Textual has issues)
    launch_console_interface(console)


def launch_beautiful_interface(console: Console):
    """Launch beautiful Textual interface"""
    # For now, fall back to console since Textual has issues
    console.print("⚠️ Beautiful interface under development, using enhanced console", style="yellow")
    launch_console_interface(console)


def launch_console_interface(console: Console):
    """Launch enhanced console interface"""
    try:
        from ai_assistant import AIFileAssistant
        assistant = AIFileAssistant()
        assistant.run_interactive()
        
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="orange1")
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        console.print("\n🔧 Running basic test...", style="yellow")
        test_filesystem_manager()


def test_filesystem_manager():
    """Test basic functionality"""
    console = Console()
    console.print("\n🧪 Testing FileSystemManager...")
    
    try:
        workspace_path = Path("workspace")
        workspace_path.mkdir(exist_ok=True)
        
        from tools.file_fol_tools import FileSystemManager
        fs_manager = FileSystemManager(base_dir=str(workspace_path.resolve()))
        
        structure = fs_manager.folder_structure(".")
        console.print("✅ FileSystemManager working")
        console.print(f"📂 Workspace structure: {structure}")
        
        # Pretty print structure
        def print_structure(struct, indent=0):
            for key, value in struct.items():
                if key == '_files':
                    for file in value:
                        console.print("  " * indent + f"📄 {file}")
                else:
                    console.print("  " * indent + f"📁 {key}/")
                    if isinstance(value, dict):
                        print_structure(value, indent + 1)
        
        console.print(f"\n📁 Workspace contents ({workspace_path.resolve()}):")
        print_structure(structure)
        
        console.print(f"\n✅ Found {len(structure)} items in workspace")
        console.print("\n💡 To use full AI features:")
        console.print("   pip install -r requirements.txt")
        
    except Exception as e:
        console.print(f"❌ Test failed: {e}", style="red")


if __name__ == "__main__":
    main()

def test_filesystem_manager():
    """Test the FileSystemManager directly"""
    console = Console()
    console.print("\n🧪 Testing FileSystemManager...")
    
    try:
        # Ensure workspace exists
        workspace_path = Path("workspace")
        workspace_path.mkdir(exist_ok=True)
        
        # Create an instance of FileSystemManager with workspace restriction
        from tools.file_fol_tools import FileSystemManager
        fs_manager = FileSystemManager(base_dir=str(workspace_path.resolve()))
        
        # Test folder structure
        structure = fs_manager.folder_structure(".")
        console.print("✅ FileSystemManager working correctly")
        console.print(f"📂 Workspace structure: {structure}")
        
    except Exception as e:
        console.print(f"❌ FileSystemManager test failed: {e}", style="red")
    
    # Test folder structure
    print(f"\n📁 Workspace directory structure ({workspace_path.resolve()}):")
    result = fs_manager.folder_structure()
    
    # Pretty print the structure
    def print_structure(structure, indent=0):
        for key, value in structure.items():
            if key == '_files':
                for file in value:
                    print("  " * indent + f"📄 {file}")
            else:
                print("  " * indent + f"📁 {key}/")
                if isinstance(value, dict):
                    print_structure(value, indent + 1)
    
    print_structure(result)
    
    # Test other operations
    print(f"\n✅ FileSystemManager is working correctly!")
    print(f"Found {len(result)} top-level items in workspace")
    print(f"\n💡 To use the full AI assistant, ensure all dependencies are installed:")
    print(f"   pip install -r requirements.txt")

if __name__ == "__main__":
    main()