# ask.py
# The finale: retrieve relevant chunks, then have Gemini write a grounded answer.

import os
import chromadb
from dotenv import load_dotenv
from google import genai

# --- 1. Load the secret API key from .env file ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- 2. Connect to Gemini and to the previous database  ---
gemini = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="my_docs")

# --- 3. Retrieve the most relevant chunks for the question ---
question = "Why does chunk size matter?"
results = collection.query(query_texts=[question], n_results=3)
retrieved_chunks = results["documents"][0]

# --- 4. Build the prompt: hand Gemini the chunks plus the question ---
context = "\n\n".join(retrieved_chunks)

prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

# --- 5. Ask Gemini and print the grounded answer ---
response = gemini.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt,
)

print(f"Question: {question}\n")
print("Answer:")
print(response.text)