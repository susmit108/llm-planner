from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from medical_graph import (
    ARTIFACT_DIR,
    GroqConfigurationError,
    MedQAGraphAssistant,
    MissingDependencyError,
)
from utils import load_json, save_json


CONV_FILE = Path("conversation.json")


def scroll_to_latest_message() -> None:
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        const chatInput = parentDoc.querySelector('[data-testid="stChatInput"]');
        if (chatInput) {
            chatInput.scrollIntoView({behavior: "smooth", block: "end"});
        } else {
            window.parent.scrollTo({top: parentDoc.body.scrollHeight, behavior: "smooth"});
        }
        </script>
        """,
        height=0,
    )


def load_messages() -> list[dict[str, str]]:
    payload = load_json(CONV_FILE, default={"Conversation": {}})
    conversation = payload.get("Conversation", {})
    messages: list[dict[str, str]] = []
    for _, entry in sorted(conversation.items(), key=lambda item: int(item[0])):
        if isinstance(entry, dict):
            role = entry.get("role", "assistant")
            content = entry.get("content", "")
        else:
            role = "user"
            content = str(entry)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def save_messages(messages: list[dict[str, str]]) -> None:
    conversation = {
        str(index): {"role": message["role"], "content": message["content"]}
        for index, message in enumerate(messages, start=1)
    }
    save_json(CONV_FILE, {"Conversation": conversation})


@st.cache_resource(show_spinner=False)
def load_assistant() -> tuple[MedQAGraphAssistant | None, str | None]:
    if not MedQAGraphAssistant.artifacts_ready():
        return None, "Model artifacts are not ready yet."
    try:
        assistant = MedQAGraphAssistant(artifact_dir=ARTIFACT_DIR)
        assistant.require_groq()
        return assistant, None
    except (MissingDependencyError, GroqConfigurationError) as exc:
        return None, str(exc)


def render_sidebar(assistant: MedQAGraphAssistant | None, setup_error: str | None) -> None:
    st.sidebar.title("MedQA Setup")
    st.sidebar.write("Artifacts directory:")
    st.sidebar.code(str(ARTIFACT_DIR))

    if assistant is None:
        st.sidebar.error(setup_error or "Setup is incomplete.")
        st.sidebar.markdown(
            "\n".join(
                [
                    "1. `python3 -m venv .venv`",
                    "2. `.venv/bin/pip install -r requirements.txt`",
                    "3. `.venv/bin/python train_medqa_graph.py --epochs 10`",
                    "4. Add `GROQ_API_KEY` to `.env`",
                    "5. `.venv/bin/streamlit run app.py`",
                ]
            )
        )
        return

    st.sidebar.success("Model and retrieval index loaded.")
    st.sidebar.write("Response layer: Groq required and enabled")
    if assistant.training_metrics:
        top1 = assistant.training_metrics.get("top1_accuracy", 0.0)
        top3 = assistant.training_metrics.get("top3_accuracy", 0.0)
        if top1 > 0 or top3 > 0:
            st.sidebar.write(f"Top-1 accuracy: {top1:.2%}")
            st.sidebar.write(f"Top-3 accuracy: {top3:.2%}")
        else:
            st.sidebar.caption(
                "Current artifacts were built from a short smoke test. Train for more epochs for stronger results."
            )


def render_retrieval_panel(result: dict[str, Any]) -> None:
    predictions = result.get("predictions", [])
    retrieved_cases = result.get("retrieved_cases", [])
    matched_symptoms = result.get("matched_symptoms", [])

    with st.container(border=True):
        st.markdown("**Grounding Summary**")
        if predictions:
            top_prediction = predictions[0]
            st.write(f"Graph top action: {top_prediction['text']}")
        else:
            st.write("Graph top action: none")

        if matched_symptoms:
            st.write("Matched symptom phrases: " + ", ".join(matched_symptoms[:6]))
        else:
            st.write("Matched symptom phrases: none")

        st.write(f"Retrieved MedQA references: {len(retrieved_cases)}")
        st.caption("The final answer is written by Groq using the graph and retrieval evidence below.")

    with st.expander("Evidence"):
        if predictions:
            st.write("Ranked actions")
            for prediction in predictions:
                st.write(
                    f"- {prediction['text']} ({prediction['source']}, score={prediction['score']:.4f})"
                )
        if retrieved_cases:
            st.write("Retrieved MedQA cases")
            for case in retrieved_cases:
                st.write(
                    f"- [{case['meta_info']}] {case['question']} -> {case['answer']} "
                    f"(score={case['score']:.4f})"
                )
                if case["symptoms"]:
                    st.caption(", ".join(case["symptoms"]))


st.set_page_config(page_title="MedQA Graph Chatbot", layout="wide")
st.title("MedQA Graph Chatbot")
st.caption(
    "Symptom-aware retrieval plus a graph neural network trained on MedQA-USMLE."
)
st.info(
    "This assistant is for research and prototyping only. MedQA answers can be diagnoses, "
    "tests, treatments, or next-step actions and should not replace clinical judgment."
)

assistant, setup_error = load_assistant()
render_sidebar(assistant, setup_error)

if "messages" not in st.session_state:
    st.session_state.messages = load_messages()
if "scroll_to_latest" not in st.session_state:
    st.session_state.scroll_to_latest = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Describe symptoms or ask for the next best action")

if prompt:
    st.session_state.scroll_to_latest = True
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    save_messages(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if assistant is None:
            reply = (
                setup_error
                or "Setup is incomplete. Install dependencies, train the graph model, add `GROQ_API_KEY`, and restart Streamlit."
            )
            st.markdown(reply)
            assistant_message = {"role": "assistant", "content": reply}
        else:
            try:
                result = assistant.answer_question(prompt)
            except GroqConfigurationError as exc:
                st.error(str(exc))
                assistant_message = {"role": "assistant", "content": str(exc)}
            else:
                st.markdown(result["response"])
                st.caption("Balanced with Groq on top of graph+RAG retrieval.")
                render_retrieval_panel(result)
                assistant_message = {"role": "assistant", "content": result["response"]}

    st.session_state.messages.append(assistant_message)
    save_messages(st.session_state.messages)

if st.session_state.scroll_to_latest:
    scroll_to_latest_message()
    st.session_state.scroll_to_latest = False
