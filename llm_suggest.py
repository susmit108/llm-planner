
import os
import json
import copy
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

HISTORY_FILE = "persona_history.json"
CONVERSATION_FILE = "conversation.json"
PERSONA_FILE = "persona.json"
TASK_FILE = "task.json"

# ── Hardcoded rule descriptions so the LLM knows what's already covered ────────
EXISTING_RULE_DESCRIPTIONS = [
    "If sleep < 8 hours AND has high blood pressure → suggest more sleep to reduce BP",
    "If not exercising AND weight > 90 kg → suggest more exercise to reduce weight",
]


# ── Persona history helpers ─────────────────────────────────────────────────────

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
    history = history[-10:]  # rolling window
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
    return history


def build_state_delta(history: list[dict]) -> str:
    if len(history) < 2:
        return "Only one persona snapshot exists — no trend data yet."

    lines = ["Persona attribute changes over time (oldest → newest):"]
    keys = [k for k in history[0].keys() if not k.startswith("__")]

    for key in keys:
        values = []
        for snap in history:
            v = snap.get(key, "N/A")
            ts = snap.get("__timestamp__", "?")
            values.append(f"{ts}: {v}")
        # Only report if there is any change
        raw_vals = [snap.get(key) for snap in history]
        if len(set(str(v) for v in raw_vals)) > 1:
            lines.append(f"\n  [{key}]")
            for entry in values:
                lines.append(f"    {entry}")

    if len(lines) == 1:
        return "No attribute changes detected across snapshots yet."
    return "\n".join(lines)


# ── Core LLM suggestion function ───────────────────────────────────────────────

def suggest_tasks_via_llm(
    persona: dict,
    conversation: dict,
    existing_tasks: dict,
    history: list[dict],
    n_suggestions: int = 1,
) -> list[str]:
    """
    Call Groq LLM to suggest novel tasks based on:
      - Current persona state
      - Conversation history
      - State transitions over time (trends)
      - Tasks already generated (both rule-based and prior LLM suggestions)

    Returns a list of suggestion strings.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Build context blocks
    persona_str = json.dumps(
        {k: v for k, v in persona.items() if not k.startswith("__")},
        indent=2
    )

    conv_entries = list(conversation.get("Conversation", {}).values())
    conversation_str = "\n".join(
        f"  [{i+1}] {c}" for i, c in enumerate(conv_entries)
    ) or "No conversation entries yet."

    existing_tasks_str = "\n".join(
        f"  - {v}" for v in existing_tasks.values()
    ) or "None yet."

    existing_rules_str = "\n".join(
        f"  - {r}" for r in EXISTING_RULE_DESCRIPTIONS
    )

    state_delta_str = build_state_delta(history)

    system_prompt = (
        "You are a proactive health and lifestyle planning assistant. "
        "Your job is to look beyond hardcoded rules and spot non-obvious "
        "patterns across a user's conversation history and changing health "
        "attributes over time, then suggest personalised actionable tasks. "
        "Be concrete, specific, and kind. Never repeat tasks the user already has."
    )

    user_prompt = f"""
=== EXISTING RULE-BASED TASKS (already generated — do NOT duplicate) ===
{existing_rules_str}

=== TASKS ALREADY IN THE USER'S LIST ===
{existing_tasks_str}

=== CURRENT PERSONA / HEALTH PROFILE ===
{persona_str}

=== FULL CONVERSATION HISTORY ===
{conversation_str}

=== STATE TRANSITIONS ACROSS SESSIONS ===
{state_delta_str}

---

Your task:
Identify patterns or signals in the conversation and state transitions that the
hardcoded rules would MISS, and suggest exactly {n_suggestions} novel, specific,
actionable tasks for the user.

Guidelines:
- Look for correlations the rules ignore (e.g. stress + sleep, mood + diet, etc.)
- If state transitions show worsening trends, factor that into urgency
- Each task must be concrete and actionable (not vague like "be healthy")
- Do NOT suggest tasks already in the user's list
- Return ONLY a JSON array of strings, no extra text, no markdown fences.

Example output format:
["Task one", "Task two", "Task three"]
""".strip()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
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
        # Fallback: split on newlines if the model returned plain text
        return [
            line.strip().lstrip("-•1234567890. ")
            for line in raw.splitlines()
            if line.strip()
        ][:n_suggestions]

    return []


def run_llm_suggestions(n_suggestions: int = 1) -> list[str]:

    with open(PERSONA_FILE, "r") as f:
        persona = json.load(f)

    with open(CONVERSATION_FILE, "r") as f:
        conversation = json.load(f)

    with open(TASK_FILE, "r") as f:
        tasks = json.load(f)

    # Update rolling history BEFORE calling LLM so current state is included
    history = append_to_history(persona)

    # Only suggest if there is at least one conversation entry
    if not conversation.get("Conversation"):
        return []

    suggestions = suggest_tasks_via_llm(
        persona=persona,
        conversation=conversation,
        existing_tasks=tasks,
        history=history,
        n_suggestions=n_suggestions,
    )

    start_idx = len(tasks)
    for i, suggestion in enumerate(suggestions):
        tasks[str(start_idx + i + 1)] = f"[LLM] {suggestion}"

    with open(TASK_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

    return suggestions


if __name__ == "__main__":
    new_tasks = run_llm_suggestions()
    print("LLM-suggested tasks:")
    for t in new_tasks:
        print(f"  • {t}")