"""Parsing helpers for LLM JSON output."""
import json
import re


def extract_json(text: str) -> dict:
    """Robustly extract the first valid JSON object from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {}


def parse_decision(parsed: dict) -> int:
    """Convert final_decision string to binary label (1=Present, 0=Absent)."""
    decision = str(parsed.get("final_decision", "")).strip().lower()
    if "present" in decision or "lonely" in decision:
        return 1
    return 0
