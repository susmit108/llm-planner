
import os
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# ── file paths (same as the rest of the project) ──────────────────────────────
CONV_FILE    = "conversation.json"
TASK_FILE    = "task.json"
PERSONA_FILE = "persona.json"
LOG_FILE     = "batch_run.log"

SEPARATOR = "─" * 70


# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg: str, log_fh):
    """Print to stdout and write to log file simultaneously."""
    print(msg)
    log_fh.write(msg + "\n")
    log_fh.flush()


def add_conversation_record(text: str):
    """Append a new entry to conversation.json."""
    with open(CONV_FILE, "r") as f:
        data = json.load(f)
    convo = data["Conversation"]
    new_id = str(len(convo) + 1)
    convo[new_id] = text
    with open(CONV_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return new_id


def run_write_attr() -> bool:
    """Run write_attr.py (persona extraction). Returns True on success."""
    result = subprocess.run(
        ["python", "write_attr.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def run_generate_task() -> bool:
    """Run generate_task.py (rules + LLM suggestions). Returns True on success."""
    result = subprocess.run(
        ["python", "generate_task.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def load_tasks() -> dict:
    with open(TASK_FILE, "r") as f:
        return json.load(f)


def load_persona() -> dict:
    with open(PERSONA_FILE, "r") as f:
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
        tag = "[LLM] " if v.startswith("[LLM]") else "[RULE]"
        label = v.removeprefix("[LLM] ")
        lines.append(f"  {k}. {tag} {label}")
    return "\n".join(lines)


# ── main batch loop ───────────────────────────────────────────────────────────

def run_batch(input_file: str, start_line: int = 1, delay: float = 0):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_file}")
        return

    lines = [l.strip() for l in input_path.read_text().splitlines() if l.strip()]
    total = len(lines)

    with open(LOG_FILE, "w") as log_fh:
        log(f"LLM Planner — Batch Run", log_fh)
        log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_fh)
        log(f"File    : {input_file}  ({total} entries)", log_fh)
        log(f"Starting from line {start_line}", log_fh)
        log(SEPARATOR, log_fh)

        for idx, line in enumerate(lines, start=1):
            if idx < start_line:
                continue

            log(f"\n[Entry {idx}/{total}]", log_fh)
            log(f"  Conversation: \"{line}\"", log_fh)

            # 1. Append to conversation store
            record_id = add_conversation_record(line)
            log(f"  → Saved as record #{record_id}", log_fh)

            # 2. Extract persona attributes
            log("  → Running write_attr.py ...", log_fh)
            ok, out = run_write_attr()
            if not ok:
                log(f"  ✗ write_attr.py failed:\n{out}", log_fh)
                continue
            log("  ✓ Persona updated", log_fh)

            # 3. Generate tasks (rules + LLM)
            log("  → Running generate_task.py ...", log_fh)
            ok, out = run_generate_task()
            if not ok:
                log(f"  ✗ generate_task.py failed:\n{out}", log_fh)
                continue
            log("  ✓ Tasks generated", log_fh)

            # 4. Print current state snapshot
            persona = load_persona()
            tasks   = load_tasks()

            log("\n  📋 Persona state:", log_fh)
            log(format_persona(persona), log_fh)

            log("\n  ✅ Current task list:", log_fh)
            log(format_tasks(tasks), log_fh)

            log(SEPARATOR, log_fh)

            if delay > 0 and idx < total:
                time.sleep(delay)

        log(f"\nBatch complete. {total - start_line + 1} entries processed.", log_fh)
        log(f"Log saved to: {LOG_FILE}", log_fh)

        # Final summary
        log("\n" + SEPARATOR, log_fh)
        log("FINAL TASK LIST", log_fh)
        log(SEPARATOR, log_fh)
        final_tasks = load_tasks()
        rule_tasks = {k: v for k, v in final_tasks.items() if not v.startswith("[LLM]")}
        llm_tasks  = {k: v for k, v in final_tasks.items() if v.startswith("[LLM]")}

        log(f"\nRule-based ({len(rule_tasks)}):", log_fh)
        for k, v in rule_tasks.items():
            log(f"  {k}. {v}", log_fh)

        log(f"\nLLM-suggested ({len(llm_tasks)}):", log_fh)
        for k, v in llm_tasks.items():
            log(f"  {k}. {v.removeprefix('[LLM] ')}", log_fh)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run llm-planner pipeline on a text file of conversation entries."
    )
    parser.add_argument(
        "--file", default="conversations.txt",
        help="Path to the input text file (one conversation entry per line)."
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Line number to start from (useful for resuming). Default: 1."
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between entries (helps with API rate limits). Default: 1."
    )
    args = parser.parse_args()
    run_batch(input_file=args.file, start_line=args.start, delay=args.delay)