"""Vercel serverless function entry point for Aerospace MCP HTTP API.

This module exports the FastAPI application for Vercel's Python runtime.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Import the FastAPI app from main.py

# Vercel expects the app to be named 'app' or 'handler'
# The app variable is already correctly named
