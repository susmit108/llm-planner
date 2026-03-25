import json

empty_dict = {}

with open('task.json', 'w') as f:
    json.dump(empty_dict, f, indent=4)

with open('persona.json', 'r') as f:
    d = json.load(f)

for k in list(d.keys()):
    d[k] = -1

with open('persona.json', 'w') as f:
    json.dump(d, f, indent=4)

with open('conversation.json', 'r') as f:
    d = json.load(f)

d['Conversation'] = empty_dict

with open('conversation.json', 'w') as f:
    json.dump(d, f, indent=4)

