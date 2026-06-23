"""
BharatBench Dataset Schema

Each question has:
  - id          : unique identifier
  - language    : "bengali" | "hindi" | "english"
  - category    : "math" | "reasoning" | "knowledge" | "instruction" | "code"
  - difficulty  : "easy" | "medium" | "hard"
  - question    : the prompt sent to the model
  - reference   : ground-truth answer (for scoring)
  - requires_tool: whether the question needs a tool (web/calculator/code)
"""

SCHEMA = {
    "id": str,
    "language": ["bengali", "hindi", "english"],
    "category": ["math", "reasoning", "knowledge", "instruction", "code"],
    "difficulty": ["easy", "medium", "hard"],
    "question": str,
    "reference": str,
    "requires_tool": bool,
}
