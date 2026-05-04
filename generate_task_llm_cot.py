"""
generate_task_llm_cot.py
------------------------
Create care tasks using a two-step LLM pipeline:
  1. Structured reasoning over the evolving persona and conversation history
  2. Task generation grounded in that reasoning

Usage:
    python generate_task_llm_cot.py
    python generate_task_llm_cot.py --n-suggestions 3
    python generate_task_llm_cot.py --no-refresh-persona
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


def build_reasoning_summary(
    client: Groq,
    persona: dict,
    conversation: dict,
    existing_tasks: dict,
    history: list[dict],
) -> dict:
    persona_text = json.dumps(known_persona(persona), indent=2) or "{}"
    conversation_text = build_conversation_text(conversation)
    state_delta = build_state_delta(history)
    existing_tasks_text = (
        "\n".join(f"- {task}" for task in existing_tasks.values()) or "No tasks yet."
    )

    system_prompt = (
        "You are a careful health-planning assistant. Analyze the patient's evolving "
        "persona and conversation before generating tasks. Keep reasoning concise, "
        "evidence-based, and grounded in the provided information only."
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

Return a JSON object with exactly these keys:
- "signals": short evidence-backed observations from the conversation/persona
- "trends": changes over time that matter for planning
- "care_focus": the most important planning priorities
- "avoid": task ideas that would duplicate or overreach

Each value must be an array of short strings.
Return ONLY valid JSON.
""".strip()

    response = client.chat.completions.create(
        model=get_groq_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    raw = response.choices[0].message.content or ""
    reasoning = parse_json_response(raw, dict)

    return {
        "signals": [str(item).strip() for item in reasoning.get("signals", []) if str(item).strip()],
        "trends": [str(item).strip() for item in reasoning.get("trends", []) if str(item).strip()],
        "care_focus": [str(item).strip() for item in reasoning.get("care_focus", []) if str(item).strip()],
        "avoid": [str(item).strip() for item in reasoning.get("avoid", []) if str(item).strip()],
    }


def suggest_tasks_from_reasoning(
    client: Groq,
    reasoning: dict,
    persona: dict,
    existing_tasks: dict,
    n_suggestions: int,
) -> list[str]:
    persona_text = json.dumps(known_persona(persona), indent=2) or "{}"
    reasoning_text = json.dumps(reasoning, indent=2)
    existing_tasks_text = (
        "\n".join(f"- {task}" for task in existing_tasks.values()) or "No tasks yet."
    )

    system_prompt = (
        "You convert structured health-planning reasoning into concrete tasks. "
        "Make each task specific, supportive, and directly grounded in the "
        "reasoning summary. Do not duplicate existing tasks."
    )

    user_prompt = f"""
CURRENT PERSONA
{persona_text}

STRUCTURED REASONING
{reasoning_text}

EXISTING TASKS
{existing_tasks_text}

Generate exactly {n_suggestions} new tasks.

Rules:
- Each task must be directly supported by the reasoning summary.
- Make tasks actionable and personalized.
- Avoid diagnosis claims or fear-based wording.
- Do not repeat or closely paraphrase an existing task.
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
    parser = argparse.ArgumentParser(
        description="Generate tasks with a structured LLM reasoning step before task creation."
    )
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

    reasoning = build_reasoning_summary(
        client=client,
        persona=persona,
        conversation=conversation,
        existing_tasks=tasks,
        history=history,
    )
    suggestions = suggest_tasks_from_reasoning(
        client=client,
        reasoning=reasoning,
        persona=persona,
        existing_tasks=tasks,
        n_suggestions=args.n_suggestions,
    )
    added = append_tasks_with_prefix("[LLM-COT] ", suggestions)

    if reasoning["signals"] or reasoning["trends"] or reasoning["care_focus"]:
        print("Structured reasoning summary:")
        for section in ("signals", "trends", "care_focus"):
            values = reasoning.get(section, [])
            if not values:
                continue
            print(f"- {section}:")
            for value in values:
                print(f"    • {value}")

    if not added:
        print("No new LLM+CoT tasks were added.")
        return

    print(f"Added {len(added)} LLM+CoT task(s):")
    for task in added:
        print(f"  - {task}")


if __name__ == "__main__":
    main()
