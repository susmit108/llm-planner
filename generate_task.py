import json
from rules import Rules
with open('task.json','r') as f:
    task = json.load(f)

with open('persona.json','r') as f:
    persona = json.load(f)

with open('task.json','r') as f:
    task = json.load(f)

new_task = Rules(persona=persona, task=task).task

with open('task.json','w') as f:
    json.dump(new_task, f, indent=4)




