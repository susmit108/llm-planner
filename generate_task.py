import json
from rules import Rules
from llm_suggest import run_llm_suggestions

# ── Step 1: Rule-based generation (original behaviour, untouched) ──────────────

with open("task.json", "r") as f:
    task = json.load(f)

with open("persona.json", "r") as f:
    persona = json.load(f)

new_task = Rules(persona=persona, task=task).task

with open("task.json", "w") as f:
    json.dump(new_task, f, indent=4)


# ── Step 2: LLM-based novel suggestion (new extension) ────────────────────────

llm_tasks = run_llm_suggestions(n_suggestions=1)

if llm_tasks:
    print(f"LLM suggested {len(llm_tasks)} additional task(s):")
    for t in llm_tasks:
        print(f"  • {t}")
else:
    print("LLM returned no additional suggestions (not enough conversation context yet).")