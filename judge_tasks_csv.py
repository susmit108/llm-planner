"""
judge_tasks_csv.py
------------------
LLM-as-a-judge for scoring how well each suggested task matches the user query
that triggered it.

Input:
    results/generated_tasks.csv

Output:
    results/generated_tasks_judged.csv
    results/task_judge_prompt.txt

Usage:
    ./.venv/bin/python judge_tasks_csv.py
    ./.venv/bin/python judge_tasks_csv.py --input results/generated_tasks.csv --output results/generated_tasks_judged.csv
    ./.venv/bin/python judge_tasks_csv.py --task-json task.json --conversation-json conversation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for a clinical task-suggestion pipeline.

Your job is to judge how well a suggested task matches the specific user query
that triggered it. You are not acting as a doctor giving fresh advice. You are
grading the quality of alignment between:
1. the user's utterance at a given turn, and
2. the task that the system suggested for that turn.

Primary objective:
Measure whether the task is an appropriate, relevant, and useful response to
the user's query in that turn.

What "good alignment" means:
- The task addresses information explicitly stated in the user query, or a very
  strong and reasonable implication from that query.
- The task is actionable and specific enough to be useful.
- The task is clinically and behaviorally appropriate for the information in
  the query.
- The task is not overly generic, unrelated, or dependent on facts that are not
  supported by the query itself.

Important judging rule:
Judge mainly against the current user query, not the entire patient history.
If a task could be reasonable only because of earlier turns, but is weakly
supported by the current query alone, you must penalize it.

Likert scale:
1 = Very poor match
The task is irrelevant, unsafe, unsupported, or clearly mismatched.

2 = Weak match
The task has a slight connection to the query but is mostly generic, poorly
grounded, or not the most appropriate response.

3 = Moderate match
The task is partially relevant and plausible, but it is somewhat generic,
indirect, incomplete, or only loosely tied to the query.

4 = Good match
The task is relevant, appropriate, and useful, with only minor issues in
specificity or tightness of fit.

5 = Excellent match
The task is directly responsive, well-grounded in the query, actionable,
specific, and clearly appropriate.

Scoring dimensions to consider before assigning the final score:
- Relevance to the user's current query
- Support from explicitly stated facts in the query
- Actionability and specificity
- Clinical/common-sense appropriateness
- Absence of unsupported leaps

Return ONLY valid JSON with this schema:
{
  "likert_score": 1,
  "rating_label": "very poor match",
  "short_reason": "1-2 sentence explanation grounded in the query and task."
}
""".strip()

JUDGE_FULL_CONVERSATION_SYSTEM_PROMPT = """
You are an expert evaluator for a clinical task-suggestion pipeline.

Your job is to judge how well a suggested task matches the entire patient
conversation. You are not acting as a doctor giving fresh advice. You are
grading the quality of alignment between:
1. the full conversation context, and
2. the task that the system suggested.

Primary objective:
Measure whether the task is appropriate, relevant, useful, and clinically
reasonable given the whole conversation.

What "good alignment" means:
- The task addresses information explicitly stated in the conversation, or a
  strong and reasonable implication from the conversation.
- The task is actionable and specific enough to be useful.
- The task is clinically and behaviorally appropriate for the patient's overall
  profile and stated needs.
- The task is not generic, unrelated, redundant, or based on unsupported leaps.

Likert scale:
1 = Very poor match
The task is irrelevant, unsafe, unsupported, or clearly mismatched.

2 = Weak match
The task has a slight connection to the conversation but is mostly generic,
poorly grounded, redundant, or not the most appropriate response.

3 = Moderate match
The task is partially relevant and plausible, but it is somewhat generic,
indirect, incomplete, or only loosely tied to the conversation.

4 = Good match
The task is relevant, appropriate, and useful, with only minor issues in
specificity, redundancy, or tightness of fit.

5 = Excellent match
The task is directly supported by the conversation, actionable, specific, and
clearly appropriate for the patient's overall situation.

Scoring dimensions to consider before assigning the final score:
- Relevance to the full conversation
- Support from explicitly stated patient facts
- Actionability and specificity
- Clinical/common-sense appropriateness
- Absence of unsupported leaps or unnecessary duplication

Return ONLY valid JSON with this schema:
{
  "likert_score": 1,
  "rating_label": "very poor match",
  "short_reason": "1-2 sentence explanation grounded in the conversation and task."
}
""".strip()


def build_user_prompt(row: dict[str, str]) -> str:
    return f"""
Evaluate the following pair.

USER QUERY:
{row["conversation_text"]}

SUGGESTED TASK:
{row["task_text"]}

TASK SOURCE:
{row["task_source"]}

Instructions:
- Score the task on the 1-5 Likert scale defined in the system prompt.
- Focus on whether this task is a good suggestion for this specific query.
- Penalize tasks that are generic, weakly supported, or mainly justified by
  prior conversation context instead of this query.
- Keep the explanation concise and concrete.
- Return JSON only.
""".strip()


def build_full_conversation_user_prompt(row: dict[str, str]) -> str:
    return f"""
Evaluate the following task against the entire conversation.

FULL CONVERSATION:
{row["conversation_text"]}

SUGGESTED TASK:
{row["task_text"]}

TASK SOURCE:
{row["task_source"]}

Instructions:
- Score the task on the 1-5 Likert scale defined in the system prompt.
- Focus on whether this task is well supported by the full conversation.
- Penalize generic, redundant, weakly supported, or clinically inappropriate tasks.
- Keep the explanation concise and concrete.
- Return JSON only.
""".strip()


def task_source(task_text: str) -> str:
    if task_text.startswith("[LLM-COT]"):
        return "llm_cot"
    if task_text.startswith("[LLM-ONLY]"):
        return "llm_only"
    if task_text.startswith("[LLM]"):
        return "llm"
    if task_text.startswith("[explore]"):
        return "explore"
    if task_text.startswith("[DX]"):
        return "diagnosis"
    return "rule"


def task_label(task_text: str) -> str:
    cleaned = task_text.strip()
    for prefix in ("[LLM-COT] ", "[LLM-ONLY] ", "[LLM] ", "[explore] ", "[DX] "):
        if cleaned.startswith(prefix):
            return cleaned.removeprefix(prefix).strip()
    return cleaned


LIKERT_LABELS = {
    1: "very poor match",
    2: "weak match",
    3: "moderate match",
    4: "good match",
    5: "excellent match",
}


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
    return stripped.strip()


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def normalize_jsonish_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"(?m)^json\s*", "", normalized)
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    normalized = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', normalized)
    return normalized


def build_verdict(score: int, rating_label: str | None, reason: str | None) -> dict[str, str | int]:
    clamped = max(1, min(5, int(score)))
    label = (rating_label or LIKERT_LABELS[clamped]).strip().lower()
    short_reason = (reason or "").strip() or "Model response could not be parsed cleanly."
    return {
        "likert_score": clamped,
        "rating_label": label,
        "short_reason": short_reason,
    }


def parse_json_verdict(data: dict) -> dict[str, str | int]:
    score = data.get("likert_score", data.get("score"))
    if score is None:
        raise ValueError("Missing likert_score in judge response")
    return build_verdict(
        score=int(str(score).strip()),
        rating_label=str(data.get("rating_label", "")).strip() or None,
        reason=str(data.get("short_reason", data.get("reason", ""))).strip() or None,
    )


def parse_judge_response(raw: str) -> dict[str, str | int]:
    text = strip_code_fence(raw)

    for candidate in filter(None, [text, extract_json_object(text)]):
        try:
            return parse_json_verdict(json.loads(candidate))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        try:
            return parse_json_verdict(json.loads(normalize_jsonish_text(candidate)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    score_match = re.search(r'"?likert_score"?\s*:\s*([1-5])', text, flags=re.IGNORECASE)
    if not score_match:
        score_match = re.search(r'\bscore\b\s*[:=-]?\s*([1-5])', text, flags=re.IGNORECASE)
    if not score_match:
        score_match = re.search(r'\b([1-5])\s*/\s*5\b', text, flags=re.IGNORECASE)
    if not score_match:
        raise ValueError(f"Could not parse judge response: {raw[:300]}")

    label_match = re.search(
        r'"?rating_label"?\s*:\s*"([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    reason_match = re.search(
        r'"?(?:short_reason|reason)"?\s*:\s*"([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    if not reason_match:
        lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
        reason = next(
            (
                line for line in lines
                if not re.search(r'likert_score|rating_label|\bscore\b', line, flags=re.IGNORECASE)
            ),
            "",
        )
    else:
        reason = reason_match.group(1)

    return build_verdict(
        score=int(score_match.group(1)),
        rating_label=label_match.group(1) if label_match else None,
        reason=reason,
    )


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_model() -> str:
    model = os.getenv("JUDGE_GROQ_MODEL") or os.getenv("GROQ_MODEL")
    if not model:
        raise RuntimeError("GROQ_MODEL is not set. Add it to your .env file.")
    return model


def judge_row(
    client,
    row: dict[str, str],
    full_conversation_mode: bool = False,
) -> tuple[dict[str, str | int], str]:
    system_prompt = (
        JUDGE_FULL_CONVERSATION_SYSTEM_PROMPT
        if full_conversation_mode
        else JUDGE_SYSTEM_PROMPT
    )
    user_prompt = (
        build_full_conversation_user_prompt(row)
        if full_conversation_mode
        else build_user_prompt(row)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_raw = ""
    for _attempt in range(2):
        response = client.chat.completions.create(
            model=get_model(),
            messages=messages,
            temperature=0,
            max_tokens=250,
        )
        last_raw = response.choices[0].message.content or ""
        try:
            return parse_judge_response(last_raw), last_raw
        except ValueError:
            messages.append({"role": "assistant", "content": last_raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid for automatic parsing. "
                        "Return ONLY valid JSON with keys "
                        "`likert_score`, `rating_label`, and `short_reason`."
                    ),
                }
            )

    raise ValueError(f"Could not parse judge response after retry: {last_raw[:300]}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def read_conversation_context(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Conversation context not found: {path}")

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        entries = list(data.get("Conversation", {}).values())
        return "\n".join(f"[{idx}] {text}" for idx, text in enumerate(entries, start=1))

    return path.read_text().strip()


def override_rows_conversation(rows: list[dict[str, str]], conversation_text: str) -> list[dict[str, str]]:
    return [{**row, "conversation_text": conversation_text} for row in rows]


def read_rows_from_task_json(task_path: Path, conversation_path: Path, scope: str) -> list[dict[str, str]]:
    if not task_path.exists():
        raise FileNotFoundError(f"Task JSON not found: {task_path}")
    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation JSON not found: {conversation_path}")

    tasks = json.loads(task_path.read_text())
    conversation = json.loads(conversation_path.read_text())
    entries = list(conversation.get("Conversation", {}).values())

    if scope == "latest":
        conversation_text = str(entries[-1]) if entries else ""
        turn_index = str(len(entries))
    else:
        conversation_text = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(entries, start=1))
        turn_index = "final"

    rows = []
    for task_id, raw_task_text in tasks.items():
        rows.append(
            {
                "turn_index": turn_index,
                "conversation_text": conversation_text,
                "task_id": str(task_id),
                "task_source": task_source(str(raw_task_text)),
                "task_text": task_label(str(raw_task_text)),
                "raw_task_text": str(raw_task_text),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        fieldnames = [
            "turn_index",
            "conversation_text",
            "task_id",
            "task_source",
            "task_text",
            "raw_task_text",
            "likert_score",
            "rating_label",
            "short_reason",
            "judge_error",
        ]
    else:
        fieldnames = list(rows[0].keys())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_prompt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_text = (
        "=== SYSTEM PROMPT ===\n"
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        "=== USER PROMPT TEMPLATE ===\n"
        "Evaluate the following pair.\n\n"
        "USER QUERY:\n"
        "{conversation_text}\n\n"
        "SUGGESTED TASK:\n"
        "{task_text}\n\n"
        "TASK SOURCE:\n"
        "{task_source}\n\n"
        "Instructions:\n"
        "- Score the task on the 1-5 Likert scale defined in the system prompt.\n"
        "- Focus on whether this task is a good suggestion for this specific query.\n"
        "- Penalize tasks that are generic, weakly supported, or mainly justified by\n"
        "  prior conversation context instead of this query.\n"
        "- Keep the explanation concise and concrete.\n"
        "- Return JSON only.\n"
    )
    path.write_text(prompt_text)


def run_judging(
    input_path: str,
    output_path: str,
    prompt_output_path: str,
    task_json_path: str | None = None,
    conversation_json_path: str = "conversation.json",
    conversation_scope: str = "all",
    full_conversation_path: str | None = None,
) -> tuple[int, float]:
    load_env_file()

    if task_json_path:
        rows = read_rows_from_task_json(
            Path(task_json_path),
            Path(conversation_json_path),
            conversation_scope,
        )
    else:
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        rows = read_rows(source)

    full_conversation_mode = full_conversation_path is not None
    if full_conversation_path:
        rows = override_rows_conversation(
            rows,
            read_conversation_context(Path(full_conversation_path)),
        )

    save_prompt(Path(prompt_output_path))

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The `groq` package is required to run the judge script.") from exc

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    client = Groq(api_key=api_key)
    judged_rows = []

    progress = tqdm(rows, desc="Judging tasks", unit="task")
    for row in progress:
        try:
            verdict, _raw_response = judge_row(
                client,
                row,
                full_conversation_mode=full_conversation_mode,
            )
            judged_rows.append(
                {
                    **row,
                    "likert_score": str(verdict["likert_score"]),
                    "rating_label": str(verdict["rating_label"]),
                    "short_reason": str(verdict["short_reason"]),
                    "judge_error": "",
                }
            )
        except Exception as exc:
            judged_rows.append(
                {
                    **row,
                    "likert_score": "",
                    "rating_label": "",
                    "short_reason": "",
                    "judge_error": str(exc),
                }
            )

    write_rows(Path(output_path), judged_rows)

    avg_score = 0.0
    valid_scores = [int(row["likert_score"]) for row in judged_rows if row["likert_score"]]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)

    return len(valid_scores), avg_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use an LLM judge to score task suggestions against the user query that triggered them."
    )
    parser.add_argument(
        "--input",
        default="results/generated_tasks.csv",
        help="Input tasks CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/generated_tasks_judged.csv",
        help="Output scored CSV.",
    )
    parser.add_argument(
        "--prompt-output",
        default="results/task_judge_prompt.txt",
        help="Where to save the judge prompt text.",
    )
    parser.add_argument(
        "--task-json",
        help=(
            "Judge tasks directly from task.json instead of an exported CSV. "
            "This evaluates the final task list against conversation context, "
            "not exact per-turn trigger rows."
        ),
    )
    parser.add_argument(
        "--conversation-json",
        default="conversation.json",
        help="Conversation JSON to use with --task-json.",
    )
    parser.add_argument(
        "--conversation-scope",
        choices=["all", "latest"],
        default="all",
        help="Use all conversation turns or only the latest turn with --task-json.",
    )
    parser.add_argument(
        "--full-conversation",
        help=(
            "Override each CSV row's conversation_text with a full conversation "
            "context from conversation.json or conversation.txt, and use the "
            "full-conversation judge rubric."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_rows, avg_score = run_judging(
        args.input,
        args.output,
        args.prompt_output,
        task_json_path=args.task_json,
        conversation_json_path=args.conversation_json,
        conversation_scope=args.conversation_scope,
        full_conversation_path=args.full_conversation,
    )
    print(
        f"Judged {total_rows} task(s). Average Likert score: {avg_score:.2f}. "
        f"Saved scored CSV to {args.output}"
    )


if __name__ == "__main__":
    main()
