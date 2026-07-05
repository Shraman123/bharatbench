"""
BharatBench Dataset Schema

This is a lightweight, human-readable mirror of the authoritative schema in
dataset/schema.json (a real JSON Schema, enforced by scripts/validate_dataset.py).
Keep the two in sync.

Each question has:
  - id          : unique identifier
  - language    : "bengali" | "hindi" | "english"
  - category    : "math" | "reasoning" | "knowledge" | "instruction" | "code"
  - difficulty  : "easy" | "medium" | "hard"
  - question    : the prompt sent to the model
  - reference   : ground-truth answer (for scoring)
  - requires_tool: whether the question needs a tool (web/calculator/code).
                  INFORMATIONAL ONLY -- not currently enforced by
                  eval/runner.py; see README.md#known-limitations.
  - source      : provenance of the question (required for new questions —
                  see CONTRIBUTING.md). Existing questions predate this field
                  and currently fail schema validation on it; that gap is
                  tracked, not silently patched.
"""

SCHEMA = {
    "id": str,
    "language": ["bengali", "hindi", "english"],
    "category": ["math", "reasoning", "knowledge", "instruction", "code"],
    "difficulty": ["easy", "medium", "hard"],
    "question": str,
    "reference": str,
    "requires_tool": bool,
    "source": str,
}
