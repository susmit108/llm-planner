"""
llm_task_generation_utils.py
----------------------------
Shared helpers for LLM-driven task generation scripts that rely on the
conversation-derived persona snapshot and persona history.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CONVERSATION_FILE = Path("conversation.json")
PERSONA_FILE = Path("persona.json")
PERSONA_HISTORY_FILE = Path("persona_history.json")
TASK_FILE = Path("task.json")

TASK_PREFIXES = (
    "[LLM] ",
    "[DX] ",
    "[explore] ",
    "[LLM-ONLY] ",
    "[LLM-COT] ",
)


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r") as fh:
        return json.load(fh)


def save_json(path: Path, data: Any) -> None:
    with path.open("w") as fh:
        json.dump(data, fh, indent=4)


def refresh_persona_from_conversation() -> None:
    result = subprocess.run(
        [sys.executable, "write_attr.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "write_attr.py failed"
        raise RuntimeError(details)


def append_to_history(persona: dict) -> list[dict]:
    history = load_json(PERSONA_HISTORY_FILE, [])
    snapshot = copy.deepcopy(persona)
    snapshot["__timestamp__"] = datetime.now().isoformat(timespec="seconds")
    history.append(snapshot)
    history = history[-10:]
    save_json(PERSONA_HISTORY_FILE, history)
    return history


def load_generation_context(refresh_persona: bool = True) -> tuple[dict, dict, dict, list[dict]]:
    if refresh_persona:
        refresh_persona_from_conversation()

    persona = load_json(PERSONA_FILE, {})
    conversation = load_json(CONVERSATION_FILE, {"Conversation": {}})
    tasks = load_json(TASK_FILE, {})
    history = append_to_history(persona)
    return persona, conversation, tasks, history


def build_state_delta(history: list[dict]) -> str:
    if len(history) < 2:
        return "Only one snapshot is available, so no longitudinal trend is available yet."

    keys = [key for key in history[0] if not key.startswith("__")]
    lines = ["Attribute changes over time (oldest to newest):"]

    for key in keys:
        values = [str(snapshot.get(key, "N/A")) for snapshot in history]
        if len(set(values)) == 1:
            continue
        lines.append(f"- {key}")
        for snapshot in history:
            timestamp = snapshot.get("__timestamp__", "?")
            lines.append(f"  {timestamp}: {snapshot.get(key, 'N/A')}")

    return "\n".join(lines) if len(lines) > 1 else "No attribute changes detected yet."


def build_conversation_text(conversation: dict) -> str:
    entries = list(conversation.get("Conversation", {}).values())
    return "\n".join(f"[{index}] {entry}" for index, entry in enumerate(entries, start=1)) or "No conversation yet."


def known_persona(persona: dict) -> dict:
    return {
        key: value
        for key, value in persona.items()
        if not key.startswith("__") and value not in (-1, None, "")
    }


def strip_task_prefix(task_text: str) -> str:
    cleaned = task_text.strip()
    for prefix in TASK_PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


def parse_json_response(raw: str, expected_type: type) -> Any:
    cleaned = (raw or "").strip()
    if not cleaned:
        return expected_type()

    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, expected_type):
            return parsed
    except json.JSONDecodeError:
        pass

    if expected_type is list:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    elif expected_type is dict:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    else:
        match = None

    if not match:
        return expected_type()

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, expected_type):
            return parsed
    except json.JSONDecodeError:
        return expected_type()

    return expected_type()


def append_tasks_with_prefix(prefix: str, suggestions: list[str]) -> list[str]:
    tasks = load_json(TASK_FILE, {})
    existing_labels = {strip_task_prefix(task_text).lower() for task_text in tasks.values()}

    added = []
    for suggestion in suggestions:
        label = suggestion.strip()
        if not label:
            continue
        if label.lower() in existing_labels:
            continue

        task_id = str(len(tasks) + 1)
        tasks[task_id] = f"{prefix}{label}"
        existing_labels.add(label.lower())
        added.append(label)

    save_json(TASK_FILE, tasks)
    return added
