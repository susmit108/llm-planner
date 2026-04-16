"""
write_attr.py
-------------
Reads the conversation log, retrieves the most relevant turns via FAISS,
then asks Groq to update the UserHealthProfile.

This is the llm-planner pipeline unchanged in structure, but operating over
the extended UserHealthProfile that includes maria_paper's clinical fields.

Run automatically by the Streamlit app after each conversation turn.
"""

import os
import json
import re

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from config import get_groq_model

from persona import UserHealthProfile

load_dotenv()

PERSONA_FILE = "persona.json"
CONVERSATION_FILE = "conversation.json"


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_write_attr():
    with open(PERSONA_FILE, "r") as f:
        persona = json.load(f)

    with open(CONVERSATION_FILE, "r") as f:
        conversation = json.load(f)

    conversations = list(conversation.get("Conversation", {}).values())
    if not conversations:
        return persona  # nothing to extract yet

    docs = [Document(page_content=c) for c in conversations]
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model=get_groq_model(),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    profile = UserHealthProfile(**persona)
    retrieved_docs = retriever.invoke(
        "health details, clinical measurements, lifestyle habits, symptoms"
    )
    context = "\n".join([d.page_content for d in retrieved_docs])
    allowed_fields = list(UserHealthProfile.model_fields.keys())

    prompt = f"""
Extract user health attributes from the conversation.

Conversation:
{context}

Current profile:
{json.dumps(profile.model_dump(), indent=2)}

Allowed fields:
{json.dumps(allowed_fields, indent=2)}

Rules:
- Return a JSON object containing ONLY fields explicitly mentioned in the conversation.
- Do not include fields that were not updated in this turn.
- Boolean-like int fields: 1=yes/true, 0=no/false, -1=unknown.
- Extract numeric values only (no units).
- If nothing new is mentioned, return {{}}.
- Return ONLY valid JSON. No markdown, no preamble.
"""

    response = llm.invoke(prompt)
    partial_update = _extract_json_object(response.content if hasattr(response, "content") else str(response))

    if not isinstance(partial_update, dict):
        partial_update = {}

    filtered_update = {
        key: value
        for key, value in partial_update.items()
        if key in UserHealthProfile.model_fields
    }

    merged_profile = profile.model_dump()
    merged_profile.update(filtered_update)
    profile = UserHealthProfile(**merged_profile)

    with open(PERSONA_FILE, "w") as f:
        json.dump(profile.model_dump(), f, indent=4)

    return profile.model_dump()


if __name__ == "__main__":
    result = run_write_attr()
    print(json.dumps(result, indent=2))
