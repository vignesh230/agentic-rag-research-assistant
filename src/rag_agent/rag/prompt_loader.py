"""Load and format versioned prompts from prompts/*.yaml.

Prompts live in YAML so they can be diffed, versioned, and swapped without
touching Python.  The version field is logged with every generation call so
evaluation runs are always traceable to an exact prompt revision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Resolve relative to this file so the loader works regardless of cwd.
_PROMPTS_DIR = Path(__file__).parents[3] / "prompts"


def load(name: str) -> dict[str, Any]:
    """Load a prompt YAML by name (without .yaml extension).

    Args:
        name: Prompt file stem, e.g. "naive_rag".

    Returns:
        Parsed YAML dict containing at minimum: version, system,
        user_template.

    Raises:
        FileNotFoundError: If prompts/{name}.yaml does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def format_user(name: str, **kwargs: Any) -> tuple[str, str, str]:
    """Load a prompt and render the user_template with kwargs.

    Args:
        name: Prompt file stem.
        **kwargs: Variables substituted into user_template.

    Returns:
        Tuple of (system_prompt, user_message, prompt_version).
    """
    data = load(name)
    system = data["system"].strip()
    user = data["user_template"].format(**kwargs).strip()
    version = data.get("version", "unknown")
    return system, user, version
