"""
Shared pytest configuration.

Adds backend/ to sys.path so all test files can import backend modules directly.
No API key is needed — the LLM provider is Ollama (local), and all LLM calls
in unit/evaluation tests are mocked.
"""
import sys
from pathlib import Path

# Ensure backend is importable from tests
sys.path.insert(0, str(Path(__file__).parent / "backend"))
