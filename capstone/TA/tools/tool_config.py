"""
tool_config.py — Centralized configuration for Neo4j tool query parameters.
"""

# Weight multiplier for PREREQUISITE edges when computing semantic out-degree.
PREREQUISITE_WEIGHT: float = 1.7

## LLM picks these, clamp server-side -- backbone rel query is nested UNWIND over hub pairs
MAX_HUBS_CAP: int = 50
MAX_RESULTS_CAP: int = 50
