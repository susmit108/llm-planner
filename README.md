# MarIA + LLM Planner

This project combines a Streamlit medical chatbot with a background planning
pipeline. The chatbot, MarIA, talks to the user through Groq, can call tools for
health-related actions, and uses the planner output as live context. The planner
extracts a patient profile from the conversation and generates care tasks from
deterministic rules plus LLM suggestions.

## Architecture

The main runtime is `app.py`.

On every user message:

1. The message is appended to `conversation.json`.
2. `app.py` builds extra context from `persona.json` and `task.json`.
3. `agents/maria.py` sends the message to Groq with a system prompt and tool
   definitions from `tools.py`.
4. If Groq requests tool calls, MarIA dispatches them through
   `business/action_business.py`.
5. After MarIA replies, `app.py` runs the planner pipeline:
   - `write_attr.py` updates `persona.json` from the conversation.
   - `generate_task.py` runs `rules.py` and `llm_suggest.py`.
6. The Streamlit UI refreshes with the chatbot reply, updated profile, tasks,
   and recorded actions.

## Important Files

- `app.py` - Streamlit app and integration point for chat, persona extraction,
  task generation, and sidebar state.
- `agents/maria.py` - Groq-backed MarIA agent with an in-memory per-user thread
  and tool-call loop.
- `tools.py` - Groq function/tool schemas MarIA can call.
- `business/action_business.py` - Tool-call handlers. In standalone Streamlit
  mode these save to an in-memory store; with a database session they can persist
  to application models.
- `persona.py` - Pydantic schema for the extracted health profile.
- `write_attr.py` - Retrieves relevant conversation turns and asks Groq to
  update the health profile.
- `rules.py` - Deterministic care-task rules.
- `llm_suggest.py` - LLM-generated task suggestions that avoid duplicating the
  rule-based tasks.
- `generate_task.py` - Runs rule-based task generation and LLM suggestions.
- `reset.py` - Resets local JSON state files.
- `batch.py` - Runs the same pipeline over a text file, one message per line.

## Local State Files

The app uses JSON files as local session state:

- `conversation.json` - Raw conversation entries used by the planner.
- `persona.json` - Current extracted patient profile.
- `task.json` - Current task list.
- `persona_history.json` - Recent profile snapshots used by LLM suggestions to
  detect trends.

These files are updated while the chatbot runs. Use `reset.py` when you want a
clean patient/session state.

## Environment Setup

Create a `.env` file in the project root:

```bash
GROQ_API_KEY="your Groq API key"
GROQ_MODEL="your Groq model name"
```

Example Groq models change over time, so use the model name currently available
in your Groq account.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are setting up a fresh machine and imports are missing, install the
runtime packages used by the current code:

```bash
pip install groq python-dotenv langchain-community langchain-huggingface sentence-transformers faiss-cpu pytz pydantic
```

## Run the Chatbot

Start from a clean state if needed:

```bash
python reset.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

In the UI:

- The left panel is the MarIA chat.
- The right panel shows rule-based and LLM-suggested tasks.
- The sidebar shows the extracted patient profile and recorded actions.
- The sidebar reset button clears JSON state and MarIA's in-memory thread.

## Example Conversation

Try messages that include explicit health details:

```text
I am 52 years old, weigh 94 kg, sleep around 6 hours, and have high blood pressure.
My latest BP was 145/90 and LDL was 155.
Please remind me to take my medication tomorrow at 09:00.
```

After each message, MarIA responds first, then the background planner updates the
profile and tasks.

## Batch Mode

Use `batch.py` to process a text file where each non-empty line is one user
message:

```bash
python batch.py --file conversation.txt --delay 1.0
```

Useful options:

```bash
python batch.py --file conversation.txt --start 5
python batch.py --file conversation.txt --no-maria
```

- `--start` resumes from a line number.
- `--delay` adds seconds between entries to reduce API rate-limit pressure.
- `--no-maria` runs only the planner pipeline without the chatbot tool loop.

Batch output is written to `batch_run.log`.

## How Task Generation Works

`generate_task.py` has two stages:

1. `Rules(persona, task)` adds deterministic tasks for known patterns such as
   high blood pressure, high HbA1c, high LDL, obesity, smoking, and poor sleep.
2. `run_llm_suggestions()` asks Groq for one additional non-duplicative task
   based on conversation history, the current profile, and profile trends.

LLM-generated tasks are stored in `task.json` with the `[LLM]` prefix so the UI
can separate them from deterministic rule tasks.

## Tool Calls and Recorded Actions

MarIA can call these tools:

- `save_measurement`
- `create_alarm`
- `schedule_medical_appointment`
- `save_food_data`
- `schedule_proactive_message`

In standalone Streamlit mode, these are kept in memory and shown in the sidebar.
They are cleared when the app process restarts or when `clear_memory_store()` is
called. The JSON planner state is separate and persists until `reset.py` runs.

## Troubleshooting

- `RuntimeError: GROQ_MODEL is not set` - add `GROQ_MODEL` to `.env`.
- Authentication errors - check `GROQ_API_KEY`.
- Missing imports - install the extra runtime packages listed above.
- Slow first profile update - `write_attr.py` loads HuggingFace embeddings and
  builds a local FAISS index from the conversation.
- No tasks appear - provide explicit health details in chat, then wait for the
  background planner step to complete.
- LLM suggestions repeat or accumulate - run `python reset.py` to clear
  `task.json` and `persona_history.json`.
