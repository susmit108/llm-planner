import streamlit as st
import json
import subprocess

CONV_FILE = "conversation.json"
TASK_FILE = "task.json"


def add_record(text):
    with open(CONV_FILE, "r") as f:
        data = json.load(f)

    convo = data["Conversation"]
    new_id = str(len(convo) + 1)
    convo[new_id] = text

    with open(CONV_FILE, "w") as f:
        json.dump(data, f, indent=4)


def run_pipeline():
    subprocess.run(["python", "write_attr.py"])
    subprocess.run(["python", "generate_task.py"])


def load_tasks():
    with open(TASK_FILE, "r") as f:
        return json.load(f)


st.title("LLM Planner")

record = st.text_area("Enter new conversation record")

if st.button("Add Record"):
    if record.strip():
        add_record(record)
        run_pipeline()
        st.success("Record added and pipeline executed")
    else:
        st.warning("Record cannot be empty")

st.subheader("Generated Tasks")

tasks = load_tasks()

if len(tasks) == 0:
    st.write("No tasks generated")
else:
    for k, v in tasks.items():
        st.write(f"{k}. {v}")