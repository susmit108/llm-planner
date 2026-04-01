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
    schema = UserHealthProfile.model_json_schema()

    prompt = f"""
Extract user health attributes from the conversation.

Conversation:
{context}

JSON Schema:
{json.dumps(schema, indent=2)}

Current profile:
{json.dumps(profile.model_dump(), indent=2)}

Rules:
- Only update fields if explicitly mentioned in the conversation.
- Keep existing values for all other fields.
- Boolean-like int fields: 1=yes/true, 0=no/false, -1=unknown.
- Extract numeric values only (no units).
- Return ONLY valid JSON matching the schema. No markdown, no preamble.
"""

    structured_llm = llm.with_structured_output(UserHealthProfile)
    updated = structured_llm.invoke(prompt).model_dump()
    profile = UserHealthProfile(**updated)

    with open(PERSONA_FILE, "w") as f:
        json.dump(profile.model_dump(), f, indent=4)

    return profile.model_dump()


if __name__ == "__main__":
    result = run_write_attr()
    print(json.dumps(result, indent=2))
