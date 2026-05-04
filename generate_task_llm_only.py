"""
generate_task_llm_only.py
-------------------------
Create care tasks using only an LLM, grounded in the current conversation,
the persona extracted from that conversation, and persona history over time.

Usage:
    python generate_task_llm_only.py
    python generate_task_llm_only.py --n-suggestions 3
    python generate_task_llm_only.py --no-refresh-persona
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
from groq import Groq

from config import get_groq_model
from llm_task_generation_utils import (
    append_tasks_with_prefix,
    build_conversation_text,
    build_state_delta,
    known_persona,
    load_generation_context,
    parse_json_response,
)

load_dotenv()


def suggest_tasks_via_llm(
    client: Groq,
    persona: dict,
    conversation: dict,
    existing_tasks: dict,
    history: list[dict],
    n_suggestions: int,
) -> list[str]:
    persona_text = json.dumps(known_persona(persona), indent=2) or "{}"
    conversation_text = build_conversation_text(conversation)
    state_delta = build_state_delta(history)
    existing_tasks_text = (
        "\n".join(f"- {task}" for task in existing_tasks.values()) or "No tasks yet."
    )

    system_prompt = (
        "You generate supportive, concrete health-planning tasks from a patient's "
        "ongoing conversation and evolving persona. Use only evidence present in the "
        "conversation or extracted profile. Do not invent diagnoses, do not repeat "
        "existing tasks, and keep each task actionable and personalized."
    )

    user_prompt = f"""
CURRENT PERSONA
{persona_text}

CONVERSATION HISTORY
{conversation_text}

PERSONA CHANGES OVER TIME
{state_delta}

EXISTING TASKS
{existing_tasks_text}

Instructions:
- Suggest exactly {n_suggestions} new tasks.
- Base them on the persona generated throughout the conversation.
- Prefer concrete actions over generic advice.
- Do not mention missing data unless it naturally leads to a task.
- Do not repeat or rephrase an existing task.
- Return ONLY a JSON array of strings.
""".strip()

    response = client.chat.completions.create(
        model=get_groq_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    raw = response.choices[0].message.content or ""
    parsed = parse_json_response(raw, list)
    return [str(item).strip() for item in parsed if str(item).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tasks using only the LLM pipeline.")
    parser.add_argument("--n-suggestions", type=int, default=3, help="Number of tasks to request.")
    parser.add_argument(
        "--refresh-persona",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh persona.json from conversation.json before generating tasks.",
    )
    args = parser.parse_args()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    persona, conversation, tasks, history = load_generation_context(
        refresh_persona=args.refresh_persona
    )

    if not conversation.get("Conversation"):
        print("No conversation available yet, so no tasks were generated.")
        return

    suggestions = suggest_tasks_via_llm(
        client=client,
        persona=persona,
        conversation=conversation,
        existing_tasks=tasks,
        history=history,
        n_suggestions=args.n_suggestions,
    )
    added = append_tasks_with_prefix("[LLM-ONLY] ", suggestions)

    if not added:
        print("No new LLM-only tasks were added.")
        return

    print(f"Added {len(added)} LLM-only task(s):")
    for task in added:
        print(f"  - {task}")


if __name__ == "__main__":
    main()
