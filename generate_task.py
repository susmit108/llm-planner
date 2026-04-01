"""
generate_task.py
----------------
Runs both task generation steps:
  1. Deterministic rules (Rules engine)
  2. LLM-suggested novel tasks (Groq via llm_suggest)

Called by the Streamlit app as a subprocess after each conversation turn,
exactly as in llm-planner.
"""

import json
from rules import Rules
from llm_suggest import run_llm_suggestions

with open("task.json") as f:
    task = json.load(f)

with open("persona.json") as f:
    persona = json.load(f)

new_task = Rules(persona=persona, task=task).task

with open("task.json", "w") as f:
    json.dump(new_task, f, indent=4)

print(f"Rule-based tasks: {len([v for v in new_task.values() if not v.startswith('[LLM]')])} generated")

llm_tasks = run_llm_suggestions(n_suggestions=1)

if llm_tasks:
    print(f"LLM suggested {len(llm_tasks)} additional task(s):")
    for t in llm_tasks:
        print(f"  • {t}")
else:
    print("LLM: no additional suggestions yet (needs more conversation context).")