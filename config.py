import os

from dotenv import load_dotenv

load_dotenv()

def get_groq_model() -> str:
    model = os.getenv("GROQ_MODEL")
    if not model:
        raise RuntimeError("GROQ_MODEL is not set. Add it to your .env file.")
    return model
