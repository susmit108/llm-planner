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
from llm_explore import run_llm_exploration
from diagnosis_memory import add_diagnosis_tasks, refresh_diagnosis_memory

with open("task.json") as f:
    task = json.load(f)

with open("persona.json") as f:
    persona = json.load(f)

diagnosis_memory = refresh_diagnosis_memory()
new_task = Rules(persona=persona, task=task).task
new_task = add_diagnosis_tasks(new_task, diagnosis_memory=diagnosis_memory)

with open("task.json", "w") as f:
    json.dump(new_task, f, indent=4)

non_llm_tasks = [
    value for value in new_task.values()
    if not value.startswith("[LLM]") and not value.startswith("[explore]")
]
diagnosis_tasks = [value for value in non_llm_tasks if value.startswith("[DX]")]
rule_tasks = [value for value in non_llm_tasks if not value.startswith("[DX]")]

print(f"Rule-based tasks: {len(rule_tasks)} generated")
print(f"Diagnosis-informed tasks: {len(diagnosis_tasks)} generated")

llm_tasks = run_llm_suggestions(n_suggestions=1)

if llm_tasks:
    print(f"LLM suggested {len(llm_tasks)} additional task(s):")
    for t in llm_tasks:
        print(f"  • {t}")
else:
    print("LLM: no additional suggestions yet (needs more conversation context).")

explore_tasks = run_llm_exploration(k=5, n_suggestions=2)

if explore_tasks:
    print(f"LLM exploration suggested {len(explore_tasks)} additional task(s):")
    for t in explore_tasks:
        print(f"  • {t}")
else:
    print("LLM exploration: no suggestions yet (needs a user query and persona delta).")
