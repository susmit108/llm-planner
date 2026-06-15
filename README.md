## Setting Environment

Create a ```.env ``` file and write following:

```bash

GROQ_API_KEY = "your grok api key"
GROQ_MODEL = "your Groq model name"
MARIA_GROQ_MODEL = "tool-calling Groq model name"
JUDGE_GROQ_MODEL = "judge/evaluation Groq model name"

```

`MARIA_GROQ_MODEL` is optional, but useful if your general `GROQ_MODEL` does
not support local tool/function calling. Maria's action loop requires local
tool calling support.

`JUDGE_GROQ_MODEL` is optional, but recommended for LLM-as-judge runs. Use a
regular chat model rather than `groq/compound` to avoid oversized judge
requests.




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

The main `generate_task.py` pipeline also runs an LLM exploration step after
standard `[LLM]` suggestions. It compares the latest persona delta with the
current user query, retrieves the top relevant persona parameters, records
relations/inferences in `llm_exploration.json`, and appends suggested tasks with
the `[explore]` tag.
