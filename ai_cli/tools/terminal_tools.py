"""
Terminal Tools - Safe command execution
Provides sandboxed terminal command execution with GUI support
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
    "PySide2", "PySide6", "wxPython", "kivy", "arcade", "wx",
    "gi", "gtk", "Gtk", "cv2"  # Also OpenCV with GUI
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


def check_display_available() -> Tuple[bool, str]:
    """
    Check if a graphical display is available.
    
    Returns:
        Tuple of (is_available, display_info)
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows always has a display available (unless in service mode)
        return True, "Windows desktop"
    
    elif system == "Darwin":  # macOS
        # macOS typically has display available
        return True, "macOS display"
    
    else:  # Linux/Unix
        display = os.environ.get("DISPLAY")
        wayland = os.environ.get("WAYLAND_DISPLAY")
        
        if display:
            return True, f"X11 display: {display}"
        elif wayland:
            return True, f"Wayland display: {wayland}"
        else:
            # Try to detect if we're in a graphical session
            xdg_session = os.environ.get("XDG_SESSION_TYPE")
            if xdg_session in ("x11", "wayland"):
                return True, f"XDG session: {xdg_session}"
            
            return False, "No display detected (headless/SSH)"


def get_gui_environment() -> Dict[str, str]:
    """
    Get environment variables needed for GUI applications.
    
    Returns:
        Dict with environment variables
    """
    env = os.environ.copy()
    
    system = platform.system()
    
    if system == "Linux":
        # Ensure DISPLAY is set for X11
        if "DISPLAY" not in env and os.path.exists("/tmp/.X11-unix"):
            # Try common display values
            for display in [":0", ":1", ":0.0"]:
                x_socket = f"/tmp/.X11-unix/X{display.split(':')[1].split('.')[0]}"
                if os.path.exists(x_socket):
                    env["DISPLAY"] = display
                    break
        
        # Wayland support
        if "WAYLAND_DISPLAY" not in env and "XDG_RUNTIME_DIR" in env:
            wayland_socket = os.path.join(env["XDG_RUNTIME_DIR"], "wayland-0")
            if os.path.exists(wayland_socket):
                env["WAYLAND_DISPLAY"] = "wayland-0"
        
        # XDG settings
        if "XDG_RUNTIME_DIR" not in env:
            env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    
    return env


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
        allow_gui: bool = True,  # Changed default to True - try to run GUI apps
    ) -> Dict[str, Any]:
        """
        Run Python code or script with full GUI support.
        
        Args:
            code: Python code to execute (creates temp file)
            script_path: Path to Python script
            args: Command line arguments
            timeout: Timeout in seconds (increased for GUI apps)
            allow_gui: If True, attempts to run GUI scripts with display
            
        Returns:
            Dict with execution result
        """
        if code and script_path:
            return {"error": "Provide either code or script_path, not both"}
        
        if not code and not script_path:
            return {"error": "Provide either code or script_path"}
        
        args = args or []
        gui_modules = set()
        
        if code:
            # Detect GUI imports
            gui_modules = detect_gui_imports(code)
            
            # Create script file (permanent for GUI, temp for non-GUI)
            if gui_modules:
                # Generate a meaningful filename for GUI scripts
                if "pygame" in gui_modules:
                    script_name = "pygame_app.py"
                elif "turtle" in gui_modules:
                    script_name = "turtle_app.py"
                elif any(qt in gui_modules for qt in ["PyQt5", "PyQt6", "PySide2", "PySide6"]):
                    script_name = "qt_app.py"
                elif "tkinter" in gui_modules:
                    script_name = "tkinter_app.py"
                else:
                    script_name = "gui_app.py"
                
                script_to_run = self.sandbox_path / script_name
            else:
                script_to_run = self.sandbox_path / "_temp_script.py"
            
            script_to_run.write_text(code, encoding="utf-8")
            
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
            except Exception:
                pass
        
        # Check if display is available for GUI scripts
        is_gui_script = len(gui_modules) > 0
        display_available, display_info = check_display_available()
        
        # Get proper environment for GUI apps
        env = get_gui_environment()
        
        try:
            # Build command
            cmd = [sys.executable, str(script_to_run)] + args
            
            # For GUI scripts, adjust timeout and behavior
            if is_gui_script:
                # GUI apps might need more time to start
                effective_timeout = max(timeout, 120)
                
                if not display_available:
                    # No display - inform user but still save the script
                    return {
                        "success": False,
                        "error": "No display available",
                        "is_gui": True,
                        "gui_modules": list(gui_modules),
                        "script": str(script_to_run),
                        "display_info": display_info,
                        "suggestion": f"The script uses {', '.join(gui_modules)} but no display is available.\n"
                                      f"The script has been saved to: {script_to_run}\n\n"
                                      f"To run it:\n"
                                      f"  • On a local terminal with display: python3 {script_to_run}\n"
                                      f"  • Via SSH with X forwarding: ssh -X user@host, then python3 {script_to_run}\n"
                                      f"  • In a graphical environment: just run the script",
                    }
            else:
                effective_timeout = timeout
            
            # Run the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(self.sandbox_path),
                env=env,
            )
            
            # Check for common GUI-related errors in stderr
            stderr = result.stderr
            gui_error_patterns = [
                "no display", "cannot open display", "tkinter.TclError",
                "turtle.Terminator", "pygame.error", "_tkinter.TclError",
                "could not connect to display", "DISPLAY"
            ]
            
            is_gui_error = any(p.lower() in stderr.lower() for p in gui_error_patterns)
            
            if is_gui_error and result.returncode != 0:
                return {
                    "success": False,
                    "error": "GUI display error",
                    "is_gui": True,
                    "gui_modules": list(gui_modules),
                    "script": str(script_to_run),
                    "exit_code": result.returncode,
                    "stderr": stderr,
                    "suggestion": f"Display connection failed. The script has been saved to: {script_to_run}\n\n"
                                  f"To run it manually:\n"
                                  f"  python3 {script_to_run}\n\n"
                                  f"Make sure you're running from a terminal with graphical access.",
                }
            
            return {
                "success": result.returncode == 0,
                "is_gui": is_gui_script,
                "gui_modules": list(gui_modules) if gui_modules else None,
                "script": str(script_to_run),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": stderr,
                "display_info": display_info if is_gui_script else None,
            }
            
        except subprocess.TimeoutExpired:
            # For GUI apps, timeout might mean the app is running (which is expected)
            if is_gui_script:
                return {
                    "success": True,
                    "is_gui": True,
                    "gui_modules": list(gui_modules),
                    "script": str(script_to_run),
                    "note": f"GUI application started and ran for {effective_timeout}s. "
                            f"This is normal for interactive GUI apps.\n"
                            f"The script is saved at: {script_to_run}",
                }
            else:
                return {
                    "error": f"Script timed out after {timeout} seconds",
                    "script": str(script_to_run),
                    "suggestion": "The script took too long. It might be waiting for user input or stuck in a loop.",
                }
        except Exception as e:
            return {"error": f"Script execution failed: {str(e)}"}
        finally:
            # Clean up temp script only if not a GUI script
            if code and not gui_modules:
                temp_script = self.sandbox_path / "_temp_script.py"
                if temp_script.exists():
                    try:
                        temp_script.unlink()
                    except:
                        pass
    
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
    """Run Python code with GUI support"""
    tools = TerminalTools()
    result = tools.run_python(code=code)
    
    # Handle no display available
    if result.get("error") == "No display available":
        output = [
            "🖥️ GUI Script Created",
            f"This script uses: {', '.join(result.get('gui_modules', []))}",
            "",
            f"✅ Script saved to: {result.get('script')}",
            "",
            result.get("suggestion", "Run the script in a graphical environment."),
        ]
        return "\n".join(output)
    
    # Handle GUI display error
    if result.get("error") == "GUI display error":
        output = [
            "⚠️ Display Connection Error",
            f"GUI modules: {', '.join(result.get('gui_modules', []))}",
            "",
            result.get("suggestion", "Run the script directly in a terminal with a display."),
        ]
        return "\n".join(output)
    
    # Handle other errors
    if "error" in result:
        output = [f"❌ Error: {result['error']}"]
        if result.get("suggestion"):
            output.append(f"💡 {result['suggestion']}")
        return "\n".join(output)
    
    output = []
    
    # Handle successful GUI app execution
    if result.get("is_gui"):
        if result.get("note"):
            output.append(f"🎮 {result['note']}")
        elif result.get("success"):
            output.append(f"✅ GUI application executed successfully!")
            output.append(f"   Script: {result.get('script')}")
            if result.get("display_info"):
                output.append(f"   Display: {result.get('display_info')}")
    elif result.get("success"):
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
    """Run a Python script with GUI support"""
    tools = TerminalTools()
    args_list = shlex.split(args) if args else []
    result = tools.run_python(script_path=script_path, args=args_list)
    
    # Handle no display available
    if result.get("error") == "No display available":
        output = [
            "🖥️ GUI Script Detected",
            f"This script uses: {', '.join(result.get('gui_modules', []))}",
            "",
            result.get("suggestion", "Run the script in a graphical environment."),
        ]
        return "\n".join(output)
    
    # Handle GUI display error
    if result.get("error") == "GUI display error":
        output = [
            "⚠️ Display Connection Error",
            f"GUI modules: {', '.join(result.get('gui_modules', []))}",
            "",
            result.get("suggestion", "Run the script directly in a terminal with a display."),
        ]
        return "\n".join(output)
    
    # Handle other errors
    if "error" in result:
        output = [f"❌ Error: {result['error']}"]
        if result.get("suggestion"):
            output.append(f"💡 {result['suggestion']}")
        return "\n".join(output)
    
    output = []
    
    # Handle successful GUI app execution
    if result.get("is_gui"):
        if result.get("note"):
            output.append(f"🎮 {result['note']}")
        elif result.get("success"):
            output.append(f"✅ GUI application executed successfully!")
            output.append(f"   Script: {result.get('script')}")
            if result.get("display_info"):
                output.append(f"   Display: {result.get('display_info')}")
    elif result.get("success"):
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



