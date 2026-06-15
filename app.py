"""
app.py
------
Integrated Streamlit UI combining:
  - LEFT:  Chat with Maria (agentic medical assistant, Groq tool-call loop)
  - RIGHT: Task panel (rule-based + LLM-suggested, from llm-planner pipeline)
  - SIDEBAR: Live patient persona profile + recorded actions (measurements, alarms, etc.)

Integration flow on every user message:
  1. User sends message → stored in conversation.json
  2. write_attr.py + diagnosis_memory.py refresh factual and inferred memory
  3. Maria's agentic run loop executes with the refreshed memory context
  4. generate_task.py produces rule, diagnosis-informed, and LLM tasks
  5. UI refreshes with Maria's reply + updated tasks + updated persona
"""

import json
import subprocess
import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from diagnosis_memory import format_diagnosis_context, load_diagnosis_memory

load_dotenv()

# ── File paths (llm-planner convention) ───────────────────────────────────────
CONV_FILE = "conversation.json"
TASK_FILE = "task.json"
PERSONA_FILE = "persona.json"
DIAGNOSIS_MEMORY_FILE = "diagnosis_memory.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarIA + LLM Planner",
    page_icon="🏥",
    layout="wide",
)

# ── Initialise Groq client + Maria agent ──────────────────────────────────────
@st.cache_resource
def get_maria():
    from agents.maria import Maria
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return Maria(client=client)

maria = get_maria()

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{role, content, timestamp}]
if "user_id" not in st.session_state:
    st.session_state.user_id = 0

# ── File helpers ──────────────────────────────────────────────────────────────

def load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def add_to_conversation(text: str):
    data = load_json(CONV_FILE, {"Conversation": {}})
    convo = data["Conversation"]
    convo[str(len(convo) + 1)] = text
    save_json(CONV_FILE, data)


def build_extra_context() -> str:
    """
    Build the context string injected into Maria's system prompt.
    Includes the current persona snapshot + pending task list.
    This is the bridge between llm-planner and Maria.
    """
    persona = load_json(PERSONA_FILE, {})
    tasks = load_json(TASK_FILE, {})
    diagnosis_memory = load_diagnosis_memory()

    known = {k: v for k, v in persona.items() if v not in (-1, None, "") and not k.startswith("__")}
    persona_lines = "\n".join(f"  {k}: {v}" for k, v in known.items()) or "  (no data extracted yet)"

    task_lines = "\n".join(f"  {k}. {v}" for k, v in tasks.items()) or "  (no tasks yet)"

    return (
        f"EXTRACTED HEALTH PROFILE (from conversation analysis):\n{persona_lines}\n\n"
        f"{format_diagnosis_context(diagnosis_memory)}\n\n"
        f"CURRENT CARE TASKS:\n{task_lines}"
    )


def run_context_refresh_pipeline():
    """
    Refresh factual persona memory and diagnosis hypotheses from the latest
    patient message before MarIA generates any remedies.
    """
    for script in ["write_attr.py", "diagnosis_memory.py"]:
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            st.warning(f"Pipeline warning ({script}): {result.stderr[:200]}")


def run_task_pipeline():
    """
    Generate tasks after persona + diagnosis memory have been refreshed.
    """
    result = subprocess.run(
        ["python", "generate_task.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        st.warning(f"Pipeline warning (generate_task.py): {result.stderr[:200]}")


# ── Sidebar: persona + recorded actions ───────────────────────────────────────

def render_sidebar():
    st.sidebar.header("👤 Patient Profile")
    persona = load_json(PERSONA_FILE, {})

    # Clinical fields first
    clinical_keys = [
        "Age", "Weight", "Height_cm", "BMI",
        "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
        "Glycated_Hemoglobin", "HDL", "LDL", "Framingham_Score",
        "Abdominal_Circumference_cm",
    ]
    lifestyle_keys = [
        "Average_Sleeping_Hours", "Daily_Exercise", "Has_Diabetes",
        "Has_High_Blood_Pressure", "Stress_Level", "Diet_Quality",
        "Smokes", "Alcohol_Consumption",
    ]

    st.sidebar.subheader("🔬 Clinical")
    for k in clinical_keys:
        v = persona.get(k, -1)
        icon = "✅" if v not in (-1, None, "") else "❓"
        st.sidebar.markdown(f"{icon} **{k}**: `{v}`")

    st.sidebar.subheader("🏃 Lifestyle")
    for k in lifestyle_keys:
        v = persona.get(k, -1)
        icon = "✅" if v not in (-1, None, "") else "❓"
        st.sidebar.markdown(f"{icon} **{k}**: `{v}`")

    # Recorded actions (from in-memory store)
    from business.action_business import get_memory_store
    store = get_memory_store()
    diagnosis_memory = load_json(DIAGNOSIS_MEMORY_FILE, {})

    st.sidebar.divider()
    st.sidebar.header("📋 Recorded Actions")

    if store["measurements"]:
        st.sidebar.subheader("📏 Measurements")
        for m in store["measurements"][-5:]:
            st.sidebar.markdown(f"- `{m['type']}`: {m.get('value') or m.get('text_value')}")

    if store["alarms"]:
        st.sidebar.subheader("⏰ Alarms")
        for a in store["alarms"][-5:]:
            st.sidebar.markdown(f"- {a['description']} ({a['date']} {a['time']})")

    if store["appointments"]:
        st.sidebar.subheader("🏥 Appointments")
        for ap in store["appointments"][-5:]:
            label = ap.get("doctor_name") or ap.get("exam") or ap["type"]
            st.sidebar.markdown(f"- {label} on {ap['date']}")

    if store["food_data"]:
        st.sidebar.subheader("🥗 Food Data")
        for fd in store["food_data"][-5:]:
            st.sidebar.markdown(f"- [{fd['type']}] {fd['value']}")

    if store["proactive_messages"]:
        st.sidebar.subheader("📨 Proactive Messages")
        for pm in store["proactive_messages"][-3:]:
            st.sidebar.markdown(f"- {pm['description']} ({pm['theme']})")

    st.sidebar.divider()
    st.sidebar.header("🧠 Diagnosis Memory")
    symptoms = diagnosis_memory.get("symptoms_normalized", [])
    diseases = diagnosis_memory.get("candidate_diseases", [])
    if symptoms:
        st.sidebar.markdown("**Symptoms**")
        st.sidebar.markdown(", ".join(symptoms[:8]))
    else:
        st.sidebar.caption("No symptom hypotheses yet.")

    if diseases:
        st.sidebar.markdown("**Probable diseases**")
        for disease in diseases[:3]:
            st.sidebar.markdown(f"- {disease['name']} ({disease['probability']:.0%})")

    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset Session"):
        exec(open("reset.py").read())
        maria.clear_thread(st.session_state.user_id)
        st.session_state.chat_history = []
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("🏥 MarIA + LLM Planner")
st.caption(
    "Maria — agentic medical assistant (Groq tool-call loop) "
    "**+** LLM Planner — background persona extraction & task generation"
)

col_chat, col_tasks = st.columns([3, 2])

# ── LEFT: Chat with Maria ─────────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Chat with MarIA")

    # Display conversation history
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("timestamp", "")
            with st.chat_message(role):
                st.markdown(content)
                if ts:
                    st.caption(ts)

    # Input
    user_input = st.chat_input("Send a message to MarIA…")

    if user_input and user_input.strip():
        ts = datetime.now().strftime("%H:%M")

        # Show user message immediately
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "timestamp": ts,
        })

        # 1. Save to conversation.json (feeds llm-planner pipeline)
        add_to_conversation(user_input.strip())

        # 2. Refresh persona + diagnosis memory before MarIA responds
        with st.spinner("Refreshing health memory…"):
            run_context_refresh_pipeline()

        # 3. Build context from current persona + diagnosis memory + tasks
        extra_context = build_extra_context()

        # 4. Run Maria's agentic loop (tool calls handled inside)
        with st.spinner("MarIA is thinking…"):
            reply = maria.run(
                content=user_input.strip(),
                user_id=st.session_state.user_id,
                extra_context=extra_context,
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().strftime("%H:%M"),
        })

        # 5. Generate remedies/tasks from refreshed memory
        with st.spinner("Updating health profile and tasks…"):
            run_task_pipeline()

        st.rerun()

# ── RIGHT: Task panel (llm-planner) ──────────────────────────────────────────
with col_tasks:
    tasks = load_json(TASK_FILE, {})
    rule_tasks = {
        k: v for k, v in tasks.items()
        if not v.startswith("[LLM]") and not v.startswith("[DX]") and not v.startswith("[explore]")
    }
    diagnosis_tasks = {k: v for k, v in tasks.items() if v.startswith("[DX]")}
    llm_tasks = {k: v for k, v in tasks.items() if v.startswith("[LLM]")}
    explore_tasks = {k: v for k, v in tasks.items() if v.startswith("[explore]")}

    st.subheader("📋 Rule-Based Tasks")
    st.caption("Deterministic clinical rules (llm-planner + maria_paper domain)")
    if rule_tasks:
        for k, v in rule_tasks.items():
            st.markdown(f"**{k}.** {v}")
    else:
        st.info("No rule-based tasks yet. Chat with MarIA to start building your profile.")

    st.divider()

    st.subheader("🧠 Diagnosis-Informed Tasks")
    st.caption("Knowledge-graph suggestions derived from symptom extraction and disease hypotheses")
    if diagnosis_tasks:
        for k, v in diagnosis_tasks.items():
            st.markdown(f"**{k}.** {v.removeprefix('[DX] ')}")
    else:
        st.info("Diagnosis-informed tasks appear after symptoms are detected in conversation.")

    st.divider()

    st.subheader("✨ LLM-Suggested Tasks")
    st.caption("Pattern-detected by Groq across conversation history and clinical trends")
    if llm_tasks:
        for k, v in llm_tasks.items():
            st.markdown(f"**{k}.** {v.removeprefix('[LLM] ')}")
    else:
        st.info("LLM suggestions appear after a few conversation turns.")

    st.divider()

    st.subheader("🔎 LLM Exploration Tasks")
    st.caption("Explores latest persona changes against the current query")
    if explore_tasks:
        for k, v in explore_tasks.items():
            st.markdown(f"**{k}.** {v.removeprefix('[explore] ')}")
    else:
        st.info("Exploration tasks appear after a query changes the health profile.")

    # Conversation log expander
    with st.expander("📝 Conversation log (raw)"):
        conv = load_json(CONV_FILE, {"Conversation": {}})
        for idx, text in conv["Conversation"].items():
            st.markdown(f"**[{idx}]** {text}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()
