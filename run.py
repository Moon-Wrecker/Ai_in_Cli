#!/usr/bin/env python3
"""
AI CLI v2.0 - Entry Point
Run this file to start the AI-powered terminal assistant.

Usage:
    python run.py              # Start interactive mode
    python run.py --help       # Show help
"""

import sys
from pathlib import Path

# Add ai_cli to path
sys.path.insert(0, str(Path(__file__).parent / "ai_cli"))

if __name__ == "__main__":
    from main import main
    main()

