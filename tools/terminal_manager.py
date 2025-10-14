"""
Terminal Command Manager for AI File Assistant

Provides safe terminal command execution capabilities with support for
different operating systems and shell environments.
"""

import os
import sys
import subprocess
import platform
import shlex
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import re

class TerminalManager:
    """
    Manages terminal command execution across different operating systems and shells.
    """
    
    def __init__(self, safe_mode: bool = True, allowed_commands: Optional[List[str]] = None):
        """
        Initialize the TerminalManager.
        
        Args:
            safe_mode: If True, only allows safe commands
            allowed_commands: List of allowed command prefixes when in safe mode
        """
        self.safe_mode = safe_mode
        self.os_type = platform.system().lower()
        self.shell_info = self.detect_shell()
        
        # Default safe commands (can be extended)
        self.default_safe_commands = [
            # File operations
            'ls', 'dir', 'pwd', 'cd', 'mkdir', 'touch', 'cat', 'type', 'head', 'tail',
            'find', 'where', 'which', 'tree',
            
            # System info
            'whoami', 'date', 'echo', 'hostname', 'uname', 'ver', 'systeminfo',
            'ps', 'tasklist', 'top', 'htop',
            
            # Network (read-only)
            'ping', 'nslookup', 'ipconfig', 'ifconfig', 'netstat',
            
            # Development tools
            'python', 'pip', 'node', 'npm', 'git', 'java', 'javac',
            'gcc', 'make', 'cmake',
            
            # Text processing
            'grep', 'findstr', 'sort', 'uniq', 'wc', 'cut', 'awk', 'sed'
        ]
        
        self.allowed_commands = allowed_commands or self.default_safe_commands
        
    def detect_shell(self) -> Dict[str, str]:
        """
        Detect the current shell and operating system.
        
        Returns:
            Dict with shell information
        """
        shell_info = {
            'os': self.os_type,
            'shell': 'unknown',
            'shell_path': '',
            'is_windows': self.os_type == 'windows'
        }
        
        if self.os_type == 'windows':
            # Check if we're in PowerShell or CMD
            if 'POWERSHELL_TELEMETRY_OPTOUT' in os.environ or 'PSModulePath' in os.environ:
                shell_info['shell'] = 'powershell'
                shell_info['shell_path'] = 'powershell.exe'
            else:
                shell_info['shell'] = 'cmd'
                shell_info['shell_path'] = 'cmd.exe'
        else:
            # Unix-like systems
            shell = os.environ.get('SHELL', '/bin/sh')
            shell_info['shell_path'] = shell
            shell_info['shell'] = Path(shell).name
            
        return shell_info
        
    def is_command_safe(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is safe to execute.
        
        Args:
            command: The command to check
            
        Returns:
            Tuple of (is_safe, reason)
        """
        if not self.safe_mode:
            return True, "Safe mode disabled"
            
        # Split command to get the base command
        try:
            if self.shell_info['is_windows']:
                # Windows command parsing
                parts = command.strip().split()
            else:
                # Unix command parsing
                parts = shlex.split(command)
                
            if not parts:
                return False, "Empty command"
                
            base_command = parts[0].lower()
            
            # Remove common prefixes
            base_command = base_command.replace('.exe', '').replace('./', '').replace('.\\', '')
            
            # Check against allowed commands
            for allowed in self.allowed_commands:
                if base_command.startswith(allowed.lower()):
                    return True, f"Matches allowed command: {allowed}"
                    
            # Check for dangerous patterns
            dangerous_patterns = [
                'rm -rf', 'del /f', 'format', 'fdisk', 'mkfs',
                'sudo rm', 'chmod 777', 'dd if=', '> /dev/',
                'shutdown', 'reboot', 'halt', 'poweroff',
                'curl.*|.*sh', 'wget.*|.*sh', '|sh', '|bash',
                '$(', '`', '&&', '||', ';'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return False, f"Contains dangerous pattern: {pattern}"
                    
            return False, f"Command '{base_command}' not in allowed list"
            
        except Exception as e:
            return False, f"Error parsing command: {str(e)}"
            
    def execute_command(self, command: str, working_dir: Optional[str] = None, 
                       timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a terminal command safely.
        
        Args:
            command: Command to execute
            working_dir: Working directory (optional)
            timeout: Timeout in seconds
            
        Returns:
            Dict with execution results
        """
        result = {
            'command': command,
            'success': False,
            'stdout': '',
            'stderr': '',
            'return_code': -1,
            'execution_time': 0,
            'shell_info': self.shell_info
        }
        
        # Safety check
        is_safe, reason = self.is_command_safe(command)
        if not is_safe:
            result['error'] = f"Command blocked: {reason}"
            return result
            
        try:
            import time
            start_time = time.time()
            
            # Determine shell and execution method
            if self.shell_info['is_windows']:
                if self.shell_info['shell'] == 'powershell':
                    # Execute in PowerShell
                    full_command = ['powershell.exe', '-Command', command]
                else:
                    # Execute in CMD
                    full_command = ['cmd.exe', '/c', command]
            else:
                # Execute in Unix shell
                full_command = [self.shell_info['shell_path'], '-c', command]
                
            # Execute the command
            process = subprocess.run(
                full_command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            end_time = time.time()
            
            result.update({
                'success': process.returncode == 0,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'return_code': process.returncode,
                'execution_time': end_time - start_time
            })
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Command timed out after {timeout} seconds"
        except subprocess.CalledProcessError as e:
            result.update({
                'error': f"Command failed with return code {e.returncode}",
                'return_code': e.returncode
            })
        except Exception as e:
            result['error'] = f"Execution failed: {str(e)}"
            
        return result
        
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dict with system information
        """
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'shell_info': self.shell_info,
            'current_directory': os.getcwd(),
            'environment_variables': dict(os.environ),
            'safe_mode': self.safe_mode,
            'allowed_commands': self.allowed_commands[:10] if self.safe_mode else "All commands allowed"
        }
        
        return info
        
    def suggest_command(self, intent: str) -> List[str]:
        """
        Suggest commands based on user intent.
        
        Args:
            intent: What the user wants to accomplish
            
        Returns:
            List of suggested commands
        """
        suggestions = []
        intent_lower = intent.lower()
        
        # File operations
        if 'list' in intent_lower or 'show files' in intent_lower:
            if self.shell_info['is_windows']:
                suggestions.extend(['dir', 'dir /w', 'tree'])
            else:
                suggestions.extend(['ls -la', 'ls -lh', 'tree'])
                
        elif 'current directory' in intent_lower or 'where am i' in intent_lower:
            if self.shell_info['is_windows']:
                suggestions.append('cd')
            else:
                suggestions.append('pwd')
                
        elif 'create' in intent_lower and 'folder' in intent_lower:
            suggestions.append('mkdir foldername')
            
        elif 'find' in intent_lower or 'search' in intent_lower:
            if self.shell_info['is_windows']:
                suggestions.extend(['findstr "pattern" *.txt', 'dir /s filename'])
            else:
                suggestions.extend(['find . -name "pattern"', 'grep -r "pattern" .'])
                
        # System information
        elif 'system' in intent_lower or 'info' in intent_lower:
            if self.shell_info['is_windows']:
                suggestions.extend(['systeminfo', 'ver', 'whoami'])
            else:
                suggestions.extend(['uname -a', 'whoami', 'date'])
                
        # Process information
        elif 'process' in intent_lower or 'running' in intent_lower:
            if self.shell_info['is_windows']:
                suggestions.extend(['tasklist', 'tasklist /fi "status eq running"'])
            else:
                suggestions.extend(['ps aux', 'top', 'htop'])
                
        # Network
        elif 'network' in intent_lower or 'internet' in intent_lower:
            if self.shell_info['is_windows']:
                suggestions.extend(['ipconfig', 'ping google.com'])
            else:
                suggestions.extend(['ifconfig', 'ping -c 4 google.com'])
                
        return suggestions[:5]  # Limit to top 5 suggestions
