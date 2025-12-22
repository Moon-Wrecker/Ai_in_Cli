"""
Terminal Tools - Safe command execution
Provides sandboxed terminal command execution
"""

import os
import subprocess
import shlex
import platform
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_sandbox_path
from utils.security import CommandValidator, get_command_validator


# GUI modules that require a display
GUI_MODULES = {
    "turtle", "tkinter", "pygame", "pyglet", "PyQt5", "PyQt6", 
    "PySide2", "PySide6", "wxPython", "kivy", "arcade"
}


def detect_gui_imports(code: str) -> Set[str]:
    """Detect GUI library imports in Python code"""
    found = set()
    import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
    
    for match in re.finditer(import_pattern, code):
        module = match.group(1) or match.group(2)
        if module in GUI_MODULES:
            found.add(module)
    
    return found


class TerminalTools:
    """
    Provides safe terminal command execution.
    All commands are validated and run with restrictions.
    """
    
    def __init__(self, sandbox_path: Optional[Path] = None):
        self.sandbox_path = sandbox_path or get_sandbox_path()
        self.settings = get_settings()
        self.validator = get_command_validator()
        
        # Detect shell
        self.shell = self._detect_shell()
        self.system_info = self._get_system_info()
    
    def _detect_shell(self) -> str:
        """Detect the current shell"""
        shell = os.environ.get("SHELL", "")
        if not shell:
            if platform.system() == "Windows":
                shell = "cmd"
            else:
                shell = "/bin/bash"
        return shell
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "shell": self.shell,
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "home": os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
        }
    
    def execute_command(
        self,
        command: str,
        timeout: int = 30,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a terminal command with safety checks.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            cwd: Working directory (relative to sandbox)
            
        Returns:
            Dict with execution result
        """
        # Validate command
        is_safe, risk_level, message = self.validator.validate_command(command)
        
        if not is_safe:
            return {
                "error": f"Command blocked: {message}",
                "command": command,
                "risk_level": risk_level,
            }
        
        # Determine working directory
        if cwd:
            from utils.security import PathValidator
            validator = PathValidator(self.sandbox_path)
            try:
                work_dir = validator.resolve_path(cwd)
            except Exception as e:
                return {"error": f"Invalid working directory: {e}"}
        else:
            work_dir = self.sandbox_path
        
        # Get safe environment
        env = self.validator.get_safe_environment()
        
        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
                env=env,
            )
            
            return {
                "success": True,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "cwd": str(work_dir),
                "risk_level": risk_level,
                "warning": message if risk_level == "warning" else None,
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {timeout} seconds",
                "command": command,
                "timeout": timeout,
            }
        except Exception as e:
            return {
                "error": f"Command execution failed: {str(e)}",
                "command": command,
            }
    
    def run_python(
        self,
        code: str = None,
        script_path: str = None,
        args: List[str] = None,
        timeout: int = 60,
        allow_gui: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Python code or script.
        
        Args:
            code: Python code to execute (creates temp file)
            script_path: Path to Python script
            args: Command line arguments
            timeout: Timeout in seconds
            allow_gui: If False, will warn about GUI scripts
            
        Returns:
            Dict with execution result
        """
        if code and script_path:
            return {"error": "Provide either code or script_path, not both"}
        
        if not code and not script_path:
            return {"error": "Provide either code or script_path"}
        
        args = args or []
        
        if code:
            # Check for GUI imports before running
            gui_modules = detect_gui_imports(code)
            if gui_modules and not allow_gui:
                return {
                    "error": "GUI script detected",
                    "gui_modules": list(gui_modules),
                    "suggestion": f"This script uses {', '.join(gui_modules)} which requires a graphical display. "
                                  f"To run it:\n"
                                  f"1. The script has been saved to the sandbox\n"
                                  f"2. Run it directly in your terminal: python3 <script_path>\n"
                                  f"3. Make sure you have a display available (not SSH without X forwarding)",
                    "script_saved": True,
                }
            
            # Create temporary script
            temp_script = self.sandbox_path / "_temp_script.py"
            temp_script.write_text(code, encoding="utf-8")
            script_to_run = temp_script
        else:
            from utils.security import PathValidator
            validator = PathValidator(self.sandbox_path)
            try:
                script_to_run = validator.resolve_path(script_path)
            except Exception as e:
                return {"error": f"Invalid script path: {e}"}
            
            if not script_to_run.exists():
                return {"error": f"Script not found: {script_path}"}
            
            # Check script content for GUI imports
            try:
                script_content = script_to_run.read_text()
                gui_modules = detect_gui_imports(script_content)
                if gui_modules and not allow_gui:
                    return {
                        "error": "GUI script detected",
                        "gui_modules": list(gui_modules),
                        "script": str(script_to_run),
                        "suggestion": f"This script uses {', '.join(gui_modules)} which requires a graphical display. "
                                      f"Run it directly in your terminal: python3 {script_to_run}",
                    }
            except Exception:
                pass  # If we can't read, just try to run it
        
        try:
            # Build command
            cmd = [sys.executable, str(script_to_run)] + args
            
            # Set up environment to handle display issues gracefully
            env = os.environ.copy()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.sandbox_path),
                env=env,
            )
            
            # Check for common GUI-related errors in stderr
            stderr = result.stderr
            gui_error_patterns = [
                "no display", "cannot open display", "tkinter.TclError",
                "turtle.Terminator", "pygame.error", "_tkinter.TclError"
            ]
            
            is_gui_error = any(p.lower() in stderr.lower() for p in gui_error_patterns)
            
            if is_gui_error and result.returncode != 0:
                return {
                    "success": False,
                    "error": "GUI display error",
                    "script": str(script_to_run),
                    "exit_code": result.returncode,
                    "stderr": stderr,
                    "suggestion": "This script requires a graphical display (GUI). "
                                  f"Run it directly in your terminal: python3 {script_to_run}",
                }
            
            return {
                "success": result.returncode == 0,
                "script": str(script_to_run),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": stderr,
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": f"Script timed out after {timeout} seconds",
                "script": str(script_to_run),
                "suggestion": "The script took too long. It might be waiting for user input or stuck in a loop.",
            }
        except Exception as e:
            return {"error": f"Script execution failed: {str(e)}"}
        finally:
            # Clean up temp script only if not a GUI script (keep for user to run)
            if code and temp_script.exists():
                gui_modules = detect_gui_imports(code)
                if not gui_modules:
                    temp_script.unlink()
    
    def check_command(self, command: str) -> Dict[str, Any]:
        """
        Check if a command is safe to execute.
        
        Args:
            command: Command to check
            
        Returns:
            Dict with safety analysis
        """
        is_safe, risk_level, message = self.validator.validate_command(command)
        
        return {
            "command": command,
            "is_safe": is_safe,
            "risk_level": risk_level,
            "message": message,
            "would_execute": is_safe,
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            **self.system_info,
            "sandbox_path": str(self.sandbox_path),
            "current_time": datetime.now().isoformat(),
        }
    
    def list_processes(self) -> Dict[str, Any]:
        """List running processes (safe subset)"""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["tasklist", "/FO", "CSV"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            else:
                result = subprocess.run(
                    ["ps", "aux", "--sort=-%mem"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            
            return {
                "success": True,
                "output": result.stdout,
                "process_count": len(result.stdout.strip().split("\n")) - 1,
            }
            
        except Exception as e:
            return {"error": f"Failed to list processes: {str(e)}"}
    
    def get_environment_variables(self, filter_pattern: str = None) -> Dict[str, Any]:
        """
        Get environment variables (filtered for safety).
        
        Args:
            filter_pattern: Optional pattern to filter variable names
            
        Returns:
            Dict with environment variables
        """
        import re
        
        # Sensitive patterns to exclude
        sensitive_patterns = [
            r".*KEY.*", r".*SECRET.*", r".*TOKEN.*", r".*PASSWORD.*",
            r".*CREDENTIAL.*", r".*AUTH.*", r".*PRIVATE.*",
        ]
        
        env_vars = {}
        for key, value in os.environ.items():
            # Skip sensitive variables
            is_sensitive = any(
                re.match(pattern, key, re.IGNORECASE)
                for pattern in sensitive_patterns
            )
            
            if is_sensitive:
                continue
            
            # Apply filter
            if filter_pattern:
                if not re.search(filter_pattern, key, re.IGNORECASE):
                    continue
            
            env_vars[key] = value
        
        return {
            "variables": env_vars,
            "count": len(env_vars),
            "filtered": filter_pattern is not None,
        }
    
    def suggest_commands(self, intent: str) -> Dict[str, Any]:
        """
        Suggest commands based on intent.
        
        Args:
            intent: What the user wants to do
            
        Returns:
            Dict with suggested commands
        """
        intent_lower = intent.lower()
        
        suggestions = []
        
        # File operations
        if any(word in intent_lower for word in ["list", "show", "files", "directory"]):
            suggestions.extend([
                {"command": "ls -la", "description": "List all files with details"},
                {"command": "ls -lah", "description": "List files with human-readable sizes"},
                {"command": "tree -L 2", "description": "Show directory tree (2 levels)"},
            ])
        
        # Search
        if any(word in intent_lower for word in ["find", "search", "grep"]):
            suggestions.extend([
                {"command": 'find . -name "*.py"', "description": "Find Python files"},
                {"command": 'grep -r "pattern" .', "description": "Search for pattern in files"},
                {"command": "grep -rn TODO .", "description": "Find TODO comments"},
            ])
        
        # Git
        if any(word in intent_lower for word in ["git", "version", "commit", "status"]):
            suggestions.extend([
                {"command": "git status", "description": "Show git status"},
                {"command": "git log --oneline -10", "description": "Show recent commits"},
                {"command": "git diff", "description": "Show uncommitted changes"},
            ])
        
        # Python
        if any(word in intent_lower for word in ["python", "pip", "install", "package"]):
            suggestions.extend([
                {"command": "python --version", "description": "Python version"},
                {"command": "pip list", "description": "List installed packages"},
                {"command": "pip freeze", "description": "List packages with versions"},
            ])
        
        # System
        if any(word in intent_lower for word in ["disk", "space", "memory", "system"]):
            suggestions.extend([
                {"command": "df -h", "description": "Disk usage"},
                {"command": "free -h", "description": "Memory usage"},
                {"command": "uname -a", "description": "System information"},
            ])
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                {"command": "pwd", "description": "Print working directory"},
                {"command": "ls -la", "description": "List files"},
                {"command": "cat <file>", "description": "Display file contents"},
            ]
        
        return {
            "intent": intent,
            "suggestions": suggestions,
            "note": "Use execute_command to run any of these",
        }


# Tool function wrappers for agent
def execute_command(command: str, timeout: int = 30) -> str:
    """Execute a terminal command"""
    tools = TerminalTools()
    result = tools.execute_command(command, timeout)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    output = []
    if result.get("warning"):
        output.append(f"⚠️ Warning: {result['warning']}")
    
    output.append(f"Exit code: {result['exit_code']}")
    
    if result.get("stdout"):
        output.append(f"Output:\n{result['stdout']}")
    
    if result.get("stderr"):
        output.append(f"Stderr:\n{result['stderr']}")
    
    return "\n".join(output)


def run_python_code(code: str) -> str:
    """Run Python code"""
    tools = TerminalTools()
    result = tools.run_python(code=code)
    
    # Handle GUI script detection
    if result.get("error") == "GUI script detected":
        # Save the script for the user
        script_path = tools.sandbox_path / "gui_script.py"
        script_path.write_text(code, encoding="utf-8")
        
        output = [
            "⚠️ GUI Script Detected",
            f"This script uses: {', '.join(result.get('gui_modules', []))}",
            "",
            "GUI scripts cannot run in this environment because they need a display window.",
            f"",
            f"✅ I've saved the script to: {script_path}",
            "",
            "To run this script:",
            f"  1. Open a terminal with a display",
            f"  2. Run: python3 {script_path}",
        ]
        return "\n".join(output)
    
    # Handle GUI display error
    if result.get("error") == "GUI display error":
        output = [
            "⚠️ Display Error",
            "The script tried to open a GUI window but no display is available.",
            "",
            result.get("suggestion", "Run the script directly in a terminal with a display."),
        ]
        return "\n".join(output)
    
    if "error" in result:
        output = [f"❌ Error: {result['error']}"]
        if result.get("suggestion"):
            output.append(f"💡 {result['suggestion']}")
        return "\n".join(output)
    
    output = []
    
    if result.get("success"):
        output.append("✅ Code executed successfully")
    else:
        output.append(f"⚠️ Code exited with code: {result['exit_code']}")
    
    if result.get("stdout"):
        output.append(f"\nOutput:\n{result['stdout']}")
    
    if result.get("stderr"):
        stderr = result['stderr'].strip()
        # Filter out common noise
        noise_patterns = ["telemetry", "posthog", "DeprecationWarning"]
        stderr_lines = [
            line for line in stderr.split("\n")
            if not any(noise in line.lower() for noise in noise_patterns)
        ]
        if stderr_lines:
            output.append(f"\nStderr:\n" + "\n".join(stderr_lines))
    
    return "\n".join(output)


def run_python_script(script_path: str, args: str = "") -> str:
    """Run a Python script"""
    tools = TerminalTools()
    args_list = shlex.split(args) if args else []
    result = tools.run_python(script_path=script_path, args=args_list)
    
    # Handle GUI script detection
    if result.get("error") == "GUI script detected":
        output = [
            "⚠️ GUI Script Detected",
            f"This script uses: {', '.join(result.get('gui_modules', []))}",
            "",
            "GUI scripts cannot run in this environment because they need a display window.",
            "",
            "To run this script:",
            f"  1. Open a terminal with a display",
            f"  2. Navigate to the sandbox: cd {tools.sandbox_path}",
            f"  3. Run: python3 {script_path}",
        ]
        return "\n".join(output)
    
    # Handle GUI display error
    if result.get("error") == "GUI display error":
        output = [
            "⚠️ Display Error",
            "The script tried to open a GUI window but no display is available.",
            "",
            result.get("suggestion", "Run the script directly in a terminal with a display."),
        ]
        return "\n".join(output)
    
    if "error" in result:
        output = [f"❌ Error: {result['error']}"]
        if result.get("suggestion"):
            output.append(f"💡 {result['suggestion']}")
        return "\n".join(output)
    
    output = []
    
    if result.get("success"):
        output.append("✅ Script completed successfully")
    else:
        output.append(f"⚠️ Script exited with code: {result['exit_code']}")
    
    if result.get("stdout"):
        output.append(f"\nOutput:\n{result['stdout']}")
    
    if result.get("stderr"):
        stderr = result['stderr'].strip()
        # Filter out common noise
        noise_patterns = ["telemetry", "posthog", "DeprecationWarning"]
        stderr_lines = [
            line for line in stderr.split("\n")
            if not any(noise in line.lower() for noise in noise_patterns)
        ]
        if stderr_lines:
            output.append(f"\nStderr:\n" + "\n".join(stderr_lines))
    
    return "\n".join(output)


def check_command_safety(command: str) -> str:
    """Check if a command is safe"""
    tools = TerminalTools()
    result = tools.check_command(command)
    
    status = "✅ Safe" if result["is_safe"] else "❌ Blocked"
    return f"{status}\nRisk level: {result['risk_level']}\n{result['message']}"


def get_system_info() -> str:
    """Get system information"""
    import json
    tools = TerminalTools()
    result = tools.get_system_info()
    return json.dumps(result, indent=2)



