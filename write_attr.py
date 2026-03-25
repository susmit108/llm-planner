import os
import json
from rules import Rules
from persona import UserHealthProfile
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  
api_key = os.getenv('GROQ_API_KEY')

with open('persona.json','r') as f:
    persona = json.load(f)

with open('conversation.json','r') as f:
    conversation = json.load(f)


conversations = list(conversation["Conversation"].values())



docs = [Document(page_content=c) for c in conversations]

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}
)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0
)

def extract_profile(context, current_profile, schema):

    prompt = f"""
            Extract user attributes from the conversation.

            Conversation:
            {context}

            JSON Schema:
            {schema}

            Current profile:
            {current_profile}

            Rules:
            - Only update fields if explicitly mentioned.
            - Otherwise keep existing values.
            - Return only JSON.
        """

    structured_llm = llm.with_structured_output(UserHealthProfile)
    response = structured_llm.invoke(prompt)

    return response.model_dump()

profile = UserHealthProfile(**persona)

docs = retriever.invoke("health details about the user")

context = "\n".join([d.page_content for d in docs])

schema = UserHealthProfile.model_json_schema()

updated = extract_profile(
    context,
    profile.model_dump(),
    schema
)

profile = UserHealthProfile(**updated)

print(profile.model_dump())

with open('persona.json', 'w') as json_file:
    json.dump(profile.model_dump(), json_file, indent=4)