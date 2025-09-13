"""App configuration loaded from environment (.env supported).

Expose minimal OpenAI-related settings in one place so they can be versioned
without secrets. Users can override via environment variables or a local .env.
"""

from __future__ import annotations

import os

# Model to use for chat completions (cost-efficient default)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Sampling temperature
try:
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
except ValueError:
    TEMPERATURE = 0.8
