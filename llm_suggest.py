"""
llm_suggest.py
--------------
LLM-driven task suggestion (from llm-planner), extended for the medical domain.

Key additions vs original:
- System prompt is now clinically framed (mirrors Maria's care philosophy)
- EXISTING_RULE_DESCRIPTIONS covers all 11 deterministic rules (not just 2)
- State delta includes clinical field changes, not just lifestyle fields
- The LLM is asked to reason about clinical interactions (e.g. LDL + smoking + Framingham)
"""

import os
import json
import copy
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv
from config import get_groq_model

load_dotenv()

HISTORY_FILE = "persona_history.json"
CONVERSATION_FILE = "conversation.json"
PERSONA_FILE = "persona.json"
TASK_FILE = "task.json"

EXISTING_RULE_DESCRIPTIONS = [
    # llm-planner originals
    "sleep < 8h AND high BP → suggest more sleep",
    "no exercise AND weight > 90kg → suggest exercise",
    # Clinical (maria_paper domain)
    "HbA1c > 6.5% → diabetes screening",
    "diabetic AND HbA1c > 7% → endocrinology review",
    "BMI ≥ 30 → nutritional counselling",
    "BMI 25–29.9 → balanced diet plan",
    "LDL > 130 → LDL reduction strategies",
    "HDL < 40 → increase activity to raise HDL",
    "Framingham ≥ 20% → urgent cardiology",
    "Framingham 10–19% → lifestyle modification",
    "abdominal circumference > 94cm → central obesity intervention",
    "systolic BP ≥ 140 → hypertension evaluation",
    "smoker → cessation programme",
    "high stress AND poor sleep → mindfulness/CBT referral",
]


def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def append_to_history(persona: dict) -> list[dict]:
    history = load_history()
    snapshot = copy.deepcopy(persona)
    snapshot["__timestamp__"] = datetime.now().isoformat(timespec="seconds")
    history.append(snapshot)
    history = history[-10:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
    return history


def build_state_delta(history: list[dict]) -> str:
    if len(history) < 2:
        return "Only one snapshot — no trend data yet."
    lines = ["Attribute changes over time (oldest → newest):"]
    keys = [k for k in history[0] if not k.startswith("__")]
    for key in keys:
        vals = [str(s.get(key, "N/A")) for s in history]
        if len(set(vals)) > 1:
            lines.append(f"\n  [{key}]")
            for snap in history:
                lines.append(f"    {snap.get('__timestamp__', '?')}: {snap.get(key, 'N/A')}")
    return "\n".join(lines) if len(lines) > 1 else "No attribute changes detected yet."


def suggest_tasks_via_llm(persona, conversation, existing_tasks, history, n_suggestions=1):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    persona_str = json.dumps({k: v for k, v in persona.items() if not k.startswith("__")}, indent=2)
    conv_entries = list(conversation.get("Conversation", {}).values())
    conversation_str = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(conv_entries)) or "No entries yet."
    existing_tasks_str = "\n".join(f"  - {v}" for v in existing_tasks.values()) or "None yet."
    rules_str = "\n".join(f"  - {r}" for r in EXISTING_RULE_DESCRIPTIONS)
    delta_str = build_state_delta(history)

    system_prompt = (
        "You are a proactive medical and lifestyle planning assistant working alongside MarIA, "
        "a clinical health chatbot. Deterministic rules already cover common patterns. "
        "Your job is to detect non-obvious, cross-domain clinical signals in the conversation "
        "history and health attribute trends, then suggest personalised actionable tasks that "
        "the rules would miss. Be clinically accurate, specific, and kind."
    )

    user_prompt = f"""
=== RULES ALREADY COVERED (do NOT duplicate) ===
{rules_str}

=== TASKS ALREADY IN THE USER'S LIST ===
{existing_tasks_str}

=== CURRENT HEALTH PROFILE ===
{persona_str}

=== FULL CONVERSATION HISTORY ===
{conversation_str}

=== CLINICAL ATTRIBUTE TRENDS ===
{delta_str}

---
Identify patterns the rules would MISS and suggest exactly {n_suggestions} novel,
specific, actionable tasks.

Guidelines:
- Look for clinical interactions (e.g. HbA1c worsening + poor sleep + high stress)
- Consider worsening trends (e.g. weight increasing across snapshots)
- Be concrete — not "eat better" but "reduce refined carbohydrates to help manage HbA1c trend"
- Do NOT duplicate existing tasks
- Return ONLY a JSON array of strings, no markdown, no preamble.

Example: ["Schedule a sleep study: sleep duration has dropped 3 sessions in a row alongside rising BP"]
""".strip()

    response = client.chat.completions.create(
        model=get_groq_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return [str(s).strip() for s in suggestions if s]
    except json.JSONDecodeError:
        return [
            line.strip().lstrip("-•1234567890. ")
            for line in raw.splitlines() if line.strip()
        ][:n_suggestions]
    return []


def run_llm_suggestions(n_suggestions: int = 1) -> list[str]:
    with open(PERSONA_FILE) as f:
        persona = json.load(f)
    with open(CONVERSATION_FILE) as f:
        conversation = json.load(f)
    with open(TASK_FILE) as f:
        tasks = json.load(f)

    history = append_to_history(persona)

    if not conversation.get("Conversation"):
        return []

    suggestions = suggest_tasks_via_llm(
        persona=persona,
        conversation=conversation,
        existing_tasks=tasks,
        history=history,
        n_suggestions=n_suggestions,
    )

    start = len(tasks)
    for i, s in enumerate(suggestions):
        tasks[str(start + i + 1)] = f"[LLM] {s}"

    with open(TASK_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

    return suggestions


if __name__ == "__main__":
    for t in run_llm_suggestions():
        print(f"  • {t}")
