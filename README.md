## Setting Environment

Create a ```.env ``` file and write following:

```bash

GROQ_API_KEY = "your grok api key"
GROQ_MODEL = "your Groq model name"

```




## Conversation

To reset memory, run:

```bash

python reset.py

```

To start conversation:

```bash

streamlit run app.py

```

## LLM Task Generators

To generate tasks from the conversation-derived persona using only the LLM:

```bash
python generate_task_llm_only.py
```

To generate tasks using a two-step LLM + CoT-style reasoning flow:

```bash
python generate_task_llm_cot.py
```

Both scripts:

```bash
python generate_task_llm_only.py --n-suggestions 3
python generate_task_llm_cot.py --n-suggestions 3
python generate_task_llm_only.py --no-refresh-persona
python generate_task_llm_cot.py --no-refresh-persona
```

They read from `conversation.json`, refresh `persona.json` by default via `write_attr.py`, use `persona_history.json` for trend-aware prompting, and append generated tasks to `task.json`.
