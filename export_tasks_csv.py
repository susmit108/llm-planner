"""
export_tasks_csv.py
-------------------
Run the existing planner pipeline over a text file of conversation turns and
export only newly generated tasks to a structured CSV file.

Usage:
    python export_tasks_csv.py
    python export_tasks_csv.py --file conversation.txt --output results/generated_tasks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


CONV_FILE = Path("conversation.json")
TASK_FILE = Path("task.json")
PERSONA_FILE = Path("persona.json")
PERSONA_HISTORY_FILE = Path("persona_history.json")
DIAGNOSIS_MEMORY_FILE = Path("diagnosis_memory.json")

STATE_FILES = (
    CONV_FILE,
    TASK_FILE,
    PERSONA_FILE,
    PERSONA_HISTORY_FILE,
    DIAGNOSIS_MEMORY_FILE,
)

PERSONA_DEFAULT = {
    "Age": -1,
    "Weight": -1,
    "Average_Sleeping_Hours": -1,
    "Has_Diabetes": -1,
    "Has_High_Blood_Pressure": -1,
    "Daily_Exercise": -1,
    "Height_cm": -1,
    "BMI": -1,
    "Glycated_Hemoglobin": -1,
    "Blood_Pressure_Systolic": -1,
    "Blood_Pressure_Diastolic": -1,
    "HDL": -1,
    "LDL": -1,
    "Framingham_Score": -1,
    "Abdominal_Circumference_cm": -1,
    "Stress_Level": -1,
    "Diet_Quality": -1,
    "Smokes": -1,
    "Alcohol_Consumption": -1,
}


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    with path.open("w") as fh:
        json.dump(data, fh, indent=4)


def reset_state() -> None:
    write_json(PERSONA_FILE, PERSONA_DEFAULT)
    write_json(CONV_FILE, {"Conversation": {}})
    write_json(TASK_FILE, {})
    write_json(PERSONA_HISTORY_FILE, [])
    write_json(
        DIAGNOSIS_MEMORY_FILE,
        {
            "patient_messages": [],
            "symptoms_raw": [],
            "symptoms_normalized": [],
            "unmatched_symptoms": [],
            "candidate_diseases": [],
            "recommended_tasks": [],
            "metadata": {
                "updated_at": None,
                "patient_message_count": 0,
                "extractor": "lightweight-kiddi-adapter",
            },
        },
    )


@contextmanager
def preserved_state():
    snapshot = {path: path.read_text() if path.exists() else None for path in STATE_FILES}
    try:
        yield
    finally:
        for path, content in snapshot.items():
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_text(content)


def run_script(script_name: str) -> str:
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or f"{script_name} failed without output"
        raise RuntimeError(details)
    return result.stdout.strip()


def add_conversation_record(text: str) -> str:
    data = load_json(CONV_FILE, {"Conversation": {}})
    conversation = data["Conversation"]
    new_id = str(len(conversation) + 1)
    conversation[new_id] = text
    write_json(CONV_FILE, data)
    return new_id


def load_tasks() -> dict[str, str]:
    return load_json(TASK_FILE, {})


def task_source(task_text: str) -> str:
    if task_text.startswith("[LLM]"):
        return "llm"
    if task_text.startswith("[DX]"):
        return "diagnosis"
    return "rule"


def task_label(task_text: str) -> str:
    return task_text.removeprefix("[LLM] ").removeprefix("[DX] ").strip()


def csv_json(value) -> str:
    return json.dumps(value, ensure_ascii=True)


def normalize_line(line: str) -> str:
    cleaned = line.strip().rstrip(",").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()
    return cleaned


def read_conversation_lines(path: Path) -> list[str]:
    raw_lines = path.read_text().splitlines()
    return [line for line in (normalize_line(raw) for raw in raw_lines) if line]


def export_tasks(input_file: str, output_file: str) -> tuple[int, int]:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    lines = read_conversation_lines(input_path)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with preserved_state():
        reset_state()

        progress = tqdm(
            enumerate(lines, start=1),
            total=len(lines),
            desc="Processing conversations",
            unit="turn",
        )
        for turn_index, line in progress:
            before_tasks = load_tasks()
            add_conversation_record(line)

            run_script("write_attr.py")
            run_script("diagnosis_memory.py")
            run_script("generate_task.py")

            after_tasks = load_tasks()
            diagnosis_memory = load_json(DIAGNOSIS_MEMORY_FILE, {})
            new_task_ids = [task_id for task_id in after_tasks if task_id not in before_tasks]

            for task_id in new_task_ids:
                text = after_tasks[task_id]
                rows.append(
                    {
                        "turn_index": turn_index,
                        "conversation_text": line,
                        "task_id": task_id,
                        "task_source": task_source(text),
                        "task_text": task_label(text),
                        "raw_task_text": text,
                        "symptoms_raw": csv_json(diagnosis_memory.get("symptoms_raw", [])),
                        "symptoms_normalized": csv_json(diagnosis_memory.get("symptoms_normalized", [])),
                        "unmatched_symptoms": csv_json(diagnosis_memory.get("unmatched_symptoms", [])),
                    }
                )

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "turn_index",
                "conversation_text",
                "task_id",
                "task_source",
                "task_text",
                "raw_task_text",
                "symptoms_raw",
                "symptoms_normalized",
                "unmatched_symptoms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(lines), len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export only generated planner tasks from a conversation text file into CSV."
    )
    parser.add_argument(
        "--file",
        default="conversation.txt",
        help="Input text file with one conversation turn per line.",
    )
    parser.add_argument(
        "--output",
        default="results/generated_tasks.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_turns, total_tasks = export_tasks(args.file, args.output)
    print(
        f"Exported {total_tasks} generated task(s) from {total_turns} conversation turn(s) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
