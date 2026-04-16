"""
batch.py
--------
Batch runner for the integrated MarIA + LLM Planner system.

Original behaviour (llm-planner) preserved exactly:
  - Reads a .txt file of conversation entries (one per line)
  - Per entry: appends to conversation.json → write_attr.py → diagnosis_memory.py → generate_task.py
  - Logs persona state + task list after each entry
  - Supports --start (resume) and --delay (rate limit)

Extended for Maria integration:
  - Also runs Maria's full agentic loop on each entry
  - Maria's reply is logged (tool calls are executed inline — measurements,
    alarms, appointments, food data, proactive messages all fire)
  - Maria's reply is injected back into conversation.json so write_attr.py
    can also learn from what Maria said
  - Tool call side-effects are summarised in the log per entry
  - Final summary includes recorded actions alongside tasks

Usage:
    python batch.py --file conversations.txt --delay 1.0
    python batch.py --file conversations.txt --start 5   # resume from line 5
    python batch.py --file conversations.txt --no-maria  # planner-only mode
"""

import os
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from diagnosis_memory import format_diagnosis_context, load_diagnosis_memory

load_dotenv()

# ── File paths (llm-planner convention, unchanged) ────────────────────────────
CONV_FILE    = "conversation.json"
TASK_FILE    = "task.json"
PERSONA_FILE = "persona.json"
DIAGNOSIS_MEMORY_FILE = "diagnosis_memory.json"
LOG_FILE     = "batch_run.log"

SEPARATOR = "─" * 70


# ── Logging helper (unchanged from original) ──────────────────────────────────

def log(msg: str, log_fh):
    print(msg)
    log_fh.write(msg + "\n")
    log_fh.flush()


# ── Conversation helpers (unchanged from original) ────────────────────────────

def add_conversation_record(text: str) -> str:
    with open(CONV_FILE, "r") as f:
        data = json.load(f)
    convo = data["Conversation"]
    new_id = str(len(convo) + 1)
    convo[new_id] = text
    with open(CONV_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return new_id


# ── Pipeline subprocess runners (unchanged from original) ────────────────────

def run_write_attr():
    result = subprocess.run(["python", "write_attr.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def run_diagnosis_memory():
    result = subprocess.run(["python", "diagnosis_memory.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def run_generate_task():
    result = subprocess.run(["python", "generate_task.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip()


# ── State loaders (unchanged from original) ───────────────────────────────────

def load_tasks() -> dict:
    with open(TASK_FILE) as f:
        return json.load(f)


def load_persona() -> dict:
    with open(PERSONA_FILE) as f:
        return json.load(f)


def load_diagnosis() -> dict:
    if not os.path.exists(DIAGNOSIS_MEMORY_FILE):
        return {}
    with open(DIAGNOSIS_MEMORY_FILE) as f:
        return json.load(f)


def format_persona(persona: dict) -> str:
    lines = []
    for k, v in persona.items():
        if not k.startswith("__"):
            known = "✓" if v not in (-1, None, "") else "?"
            lines.append(f"  {known} {k}: {v}")
    return "\n".join(lines)


def format_tasks(tasks: dict) -> str:
    if not tasks:
        return "  (none)"
    lines = []
    for k, v in tasks.items():
        if v.startswith("[LLM]"):
            tag = "[LLM]"
            label = v.removeprefix("[LLM] ")
        elif v.startswith("[DX]"):
            tag = "[DX]"
            label = v.removeprefix("[DX] ")
        else:
            tag = "[RULE]"
            label = v
        lines.append(f"  {k}. {tag} {label}")
    return "\n".join(lines)


# ── Maria integration helpers (new) ───────────────────────────────────────────

def get_maria_agent():
    """Lazy-load the Maria agent (avoids import at module level for --no-maria mode)."""
    from groq import Groq
    from agents.maria import Maria
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return Maria(client=client)


def build_extra_context() -> str:
    """Build the persona+task context string injected into Maria's system prompt."""
    persona = load_persona()
    tasks = load_tasks()
    diagnosis_memory = load_diagnosis_memory()
    known = {k: v for k, v in persona.items() if v not in (-1, None, "") and not k.startswith("__")}
    persona_lines = "\n".join(f"  {k}: {v}" for k, v in known.items()) or "  (no data yet)"
    task_lines = "\n".join(f"  {k}. {v}" for k, v in tasks.items()) or "  (no tasks yet)"
    return (
        f"EXTRACTED HEALTH PROFILE:\n{persona_lines}\n\n"
        f"{format_diagnosis_context(diagnosis_memory)}\n\n"
        f"CURRENT CARE TASKS:\n{task_lines}"
    )


def format_actions(store: dict) -> str:
    """Format the in-memory action store into a readable summary."""
    lines = []
    for category, records in store.items():
        if records:
            lines.append(f"  [{category.upper()}]")
            for r in records:
                if category == "measurements":
                    lines.append(f"    • {r['type']}: {r.get('value') or r.get('text_value')}")
                elif category == "alarms":
                    lines.append(f"    • {r['description']} — {r['date']} {r['time']}")
                elif category == "appointments":
                    label = r.get("doctor_name") or r.get("exam") or r["type"]
                    lines.append(f"    • {label} on {r['date']}")
                elif category == "food_data":
                    lines.append(f"    • [{r['type']}] {r['value']}")
                elif category == "proactive_messages":
                    lines.append(f"    • {r['description']} ({r['theme']}, {r['relevance']})")
    return "\n".join(lines) if lines else "  (none)"


# ── Main batch loop ───────────────────────────────────────────────────────────

def run_batch(input_file: str, start_line: int = 1, delay: float = 0, no_maria: bool = False):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_file}")
        return

    lines = [l.strip() for l in input_path.read_text().splitlines() if l.strip()]
    total = len(lines)

    # Initialise Maria once (maintains thread state across all entries)
    maria = None
    if not no_maria:
        try:
            maria = get_maria_agent()
        except Exception as e:
            print(f"WARNING: Could not initialise Maria agent ({e}). Running in planner-only mode.")
            no_maria = True

    with open(LOG_FILE, "w") as log_fh:
        log("MarIA + LLM Planner — Batch Run", log_fh)
        log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_fh)
        log(f"File    : {input_file}  ({total} entries)", log_fh)
        log(f"Mode    : {'planner-only (--no-maria)' if no_maria else 'full (Maria + planner)'}", log_fh)
        log(f"Starting from line {start_line}", log_fh)
        log(SEPARATOR, log_fh)

        for idx, line in enumerate(lines, start=1):
            if idx < start_line:
                continue

            log(f"\n[Entry {idx}/{total}]", log_fh)
            log(f"  User: \"{line}\"", log_fh)

            # ── Step 1: Append user message to conversation.json ──────────────
            record_id = add_conversation_record(line)
            log(f"  → Saved as record #{record_id}", log_fh)

            # ── Step 2: Run Maria's agentic loop (new) ────────────────────────
            log("  → Running write_attr.py ...", log_fh)
            ok, out = run_write_attr()
            if not ok:
                log(f"  ✗ write_attr.py failed:\n{out}", log_fh)
                continue
            log("  ✓ Persona updated", log_fh)

            log("  → Running diagnosis_memory.py ...", log_fh)
            ok, out = run_diagnosis_memory()
            if not ok:
                log(f"  ✗ diagnosis_memory.py failed:\n{out}", log_fh)
                continue
            log("  ✓ Diagnosis memory updated", log_fh)

            if not no_maria:
                try:
                    from business.action_business import get_memory_store, clear_memory_store
                    # Clear per-entry so we only log THIS turn's actions
                    clear_memory_store()

                    extra_context = build_extra_context()
                    maria_reply = maria.run(
                        content=line,
                        user_id=0,  # single-user batch mode
                        extra_context=extra_context,
                    )
                    log(f"  Maria: \"{maria_reply[:200]}{'...' if len(maria_reply) > 200 else ''}\"", log_fh)

                    # Log any tool call side-effects that fired
                    store = get_memory_store()
                    actions_summary = format_actions(store)
                    if actions_summary.strip() != "(none)":
                        log("  🔧 Tool calls executed:", log_fh)
                        log(actions_summary, log_fh)

                    # Append Maria's reply to conversation.json too —
                    # write_attr.py can then learn from what Maria said
                    add_conversation_record(f"[MarIA]: {maria_reply}")

                except Exception as e:
                    log(f"  ✗ Maria error: {e}", log_fh)

            # ── Step 3: Generate tasks — rules + diagnosis + LLM ─────────────
            log("  → Running generate_task.py ...", log_fh)
            ok, out = run_generate_task()
            if not ok:
                log(f"  ✗ generate_task.py failed:\n{out}", log_fh)
                continue
            log("  ✓ Tasks generated", log_fh)

            # ── Step 4: Snapshot ───────────────────────────────────────────────
            persona = load_persona()
            diagnosis = load_diagnosis()
            tasks = load_tasks()

            log("\n  📋 Persona state:", log_fh)
            log(format_persona(persona), log_fh)

            symptoms = diagnosis.get("symptoms_normalized", [])
            diseases = diagnosis.get("candidate_diseases", [])
            if symptoms or diseases:
                log("\n  🧠 Diagnosis memory:", log_fh)
                log(f"  Symptoms: {', '.join(symptoms) if symptoms else '(none)'}", log_fh)
                if diseases:
                    for item in diseases:
                        log(
                            f"  Disease hypothesis: {item['name']} "
                            f"({item['probability']:.0%})",
                            log_fh,
                        )

            log("\n  ✅ Current task list:", log_fh)
            log(format_tasks(tasks), log_fh)

            log(SEPARATOR, log_fh)

            if delay > 0 and idx < total:
                time.sleep(delay)

        # ── Final summary ──────────────────────────────────────────────────────
        log(f"\nBatch complete. {total - start_line + 1} entries processed.", log_fh)
        log(f"Log saved to: {LOG_FILE}", log_fh)

        log("\n" + SEPARATOR, log_fh)
        log("FINAL TASK LIST", log_fh)
        log(SEPARATOR, log_fh)
        final_tasks = load_tasks()
        rule_tasks = {
            k: v for k, v in final_tasks.items()
            if not v.startswith("[LLM]") and not v.startswith("[DX]")
        }
        diagnosis_tasks = {k: v for k, v in final_tasks.items() if v.startswith("[DX]")}
        llm_tasks  = {k: v for k, v in final_tasks.items() if v.startswith("[LLM]")}

        log(f"\nRule-based ({len(rule_tasks)}):", log_fh)
        for k, v in rule_tasks.items():
            log(f"  {k}. {v}", log_fh)

        log(f"\nDiagnosis-informed ({len(diagnosis_tasks)}):", log_fh)
        for k, v in diagnosis_tasks.items():
            log(f"  {k}. {v.removeprefix('[DX] ')}", log_fh)

        log(f"\nLLM-suggested ({len(llm_tasks)}):", log_fh)
        for k, v in llm_tasks.items():
            log(f"  {k}. {v.removeprefix('[LLM] ')}", log_fh)

        log("\n" + SEPARATOR, log_fh)
        log("FINAL PERSONA", log_fh)
        log(SEPARATOR, log_fh)
        log(format_persona(load_persona()), log_fh)


# ── CLI (original args preserved + --no-maria added) ─────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MarIA + LLM Planner pipeline on a text file of conversation entries."
    )
    parser.add_argument(
        "--file", default="conversations.txt",
        help="Path to input .txt file (one patient message per line)."
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Line number to start from (useful for resuming). Default: 1."
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between entries (helps with API rate limits). Default: 1."
    )
    parser.add_argument(
        "--no-maria", action="store_true",
        help="Skip Maria's agentic loop — run planner pipeline only (original llm-planner behaviour)."
    )
    args = parser.parse_args()
    run_batch(
        input_file=args.file,
        start_line=args.start,
        delay=args.delay,
        no_maria=args.no_maria,
    )
