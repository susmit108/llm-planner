"""
llm_explore.py
--------------
LLM exploration component for persona-aware task discovery.

It compares the latest persona delta with the current user query, asks the LLM
to retrieve the top-k relevant persona parameters, identifies relationships
between them, and appends exploration-derived tasks with the [explore] tag.
"""

import json
import os
from typing import Any

from groq import Groq
from groq import APIStatusError
from dotenv import load_dotenv

from config import get_groq_model
from llm_task_generation_utils import parse_json_response, strip_task_prefix

load_dotenv()

CONVERSATION_FILE = "conversation.json"
PERSONA_FILE = "persona.json"
HISTORY_FILE = "persona_history.json"
TASK_FILE = "task.json"
EXPLORATION_FILE = "llm_exploration.json"
MAX_TASKS_IN_PROMPT = 20


def _load_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def latest_user_query(conversation: dict) -> str:
    entries = list(conversation.get("Conversation", {}).values())
    return str(entries[-1]).strip() if entries else ""


def latest_persona_delta(history: list[dict], current_persona: dict) -> dict:
    if len(history) >= 2:
        previous = history[-2]
        current = history[-1]
    elif len(history) == 1:
        previous = {}
        current = history[-1]
    else:
        previous = {}
        current = current_persona

    changed = {}
    for key, value in current.items():
        if key.startswith("__"):
            continue
        if value in (-1, None, ""):
            continue
        old_value = previous.get(key, -1)
        if old_value != value:
            changed[key] = {"before": old_value, "after": value}
    return changed


def suggest_exploration_tasks(
    persona: dict,
    user_query: str,
    delta: dict,
    existing_tasks: dict,
    k: int = 5,
    n_suggestions: int = 2,
) -> list[dict]:
    if not user_query or not delta:
        return []

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    known_persona = {
        key: value
        for key, value in persona.items()
        if not key.startswith("__") and value not in (-1, None, "")
    }
    recent_tasks = list(existing_tasks.values())[-MAX_TASKS_IN_PROMPT:]
    existing_tasks_str = "\n".join(f"- {task}" for task in recent_tasks) or "None yet."

    system_prompt = (
        "You are an LLM exploration module inside a medical task planner. "
        "Your role is not to diagnose. Your role is to connect the latest user "
        "query with newly changed persona parameters, retrieve the most relevant "
        "parameters, infer possible relationships, and propose useful follow-up tasks."
    )

    user_prompt = f"""
Current user query:
{user_query}

Latest persona delta:
{json.dumps(delta, indent=2)}

Known persona parameters:
{json.dumps(known_persona, indent=2)}

Existing tasks:
{existing_tasks_str}

Retrieve the top {k} persona parameters most relevant to the current query and
latest delta, then identify potential similarities or relationships between
those parameters. Generate exactly {n_suggestions} suggested tasks.

Rules:
- Use the latest delta as strongest evidence.
- Include related current persona values when they clarify the delta.
- Inferences must be cautious and evidence-backed.
- Tasks must be concrete, supportive, and non-duplicative.
- Do not claim a confirmed diagnosis.
- Return ONLY valid JSON: an array of objects with keys
  "top_params", "relations", "inferences", and "task".
""".strip()

    try:
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=900,
        )
    except APIStatusError as exc:
        if exc.status_code == 413:
            print("LLM exploration skipped: prompt exceeded Groq request size limit.")
            return []
        raise

    raw = response.choices[0].message.content
    parsed = parse_json_response(raw, list)
    return [item for item in parsed if isinstance(item, dict) and item.get("task")]


def append_explore_tasks(explorations: list[dict]) -> list[str]:
    tasks = _load_json(TASK_FILE, {})
    existing_labels = {strip_task_prefix(task_text).lower() for task_text in tasks.values()}
    added = []

    for item in explorations:
        task = str(item.get("task", "")).strip()
        if not task or task.lower() in existing_labels:
            continue
        tasks[str(len(tasks) + 1)] = f"[explore] {task}"
        existing_labels.add(task.lower())
        added.append(task)

    _save_json(TASK_FILE, tasks)
    return added


def run_llm_exploration(k: int = 5, n_suggestions: int = 2) -> list[str]:
    persona = _load_json(PERSONA_FILE, {})
    conversation = _load_json(CONVERSATION_FILE, {"Conversation": {}})
    history = _load_json(HISTORY_FILE, [])
    tasks = _load_json(TASK_FILE, {})

    user_query = latest_user_query(conversation)
    delta = latest_persona_delta(history, persona)

    explorations = suggest_exploration_tasks(
        persona=persona,
        user_query=user_query,
        delta=delta,
        existing_tasks=tasks,
        k=k,
        n_suggestions=n_suggestions,
    )
    _save_json(
        EXPLORATION_FILE,
        {
            "current_user_query": user_query,
            "latest_persona_delta": delta,
            "explorations": explorations,
        },
    )
    return append_explore_tasks(explorations)


if __name__ == "__main__":
    for task in run_llm_exploration():
        print(f"  - {task}")
