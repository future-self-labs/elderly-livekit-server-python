"""
Prompt loading utilities for Noah companion agent.

Architecture:
- system.txt: Tiny core identity prompt (used as `instructions`, processed every turn)
- skills/*.txt: Individual skill prompts (loaded into ChatContext once at session start)

The system prompt stays small for latency. Skills are injected as conversation context
so the LLM knows its capabilities without bloating the system prompt.
"""

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent
_SKILLS_DIR = _PROMPTS_DIR / "skills"


LANGUAGE_NAMES = {
    "nl": "Dutch",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "tr": "Turkish",
}


def load_system_prompt(user_name: str, language: str = "nl") -> str:
    """Load the system prompt with user-specific substitutions.

    This is used as the `instructions` parameter on the Agent.
    Keep it tiny — it's re-processed on every turn.
    """
    lang_name = LANGUAGE_NAMES.get(language, "Dutch")
    text = (_PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()
    return text.replace("{user_name}", user_name).replace("{language}", lang_name)


def load_all_skills() -> str:
    """Load all skill files and combine them into a single context string.

    This is injected into ChatContext once at session start.
    The LLM uses it to understand its capabilities and when to activate each skill.
    """
    skills = []
    skill_files = sorted(_SKILLS_DIR.glob("*.txt"))

    for skill_file in skill_files:
        content = skill_file.read_text(encoding="utf-8").strip()
        skills.append(content)

    return "\n\n---\n\n".join(skills)


def load_skill(name: str) -> str:
    """Load a single skill file by name (without .txt extension)."""
    path = _SKILLS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Skill not found: {name}")
    return path.read_text(encoding="utf-8").strip()


def list_skills() -> list[str]:
    """List all available skill names."""
    return [f.stem for f in sorted(_SKILLS_DIR.glob("*.txt"))]
