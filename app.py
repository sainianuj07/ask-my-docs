# app.py
# A web UI for the RAG app, built with Streamlit.

import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- Just set up Gemini and the database ONCE, then reusing them ---
@st.cache_resource
def setup():
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_docs")
    return gemini, collection

gemini, collection = setup()

# --- The answer logic (using same as ask.py, packaged into a function) ---
def answer_question(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# --- The web page ---
st.title("Ask My Docs")
st.write("Ask a question and get an answer grounded in the documents.")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
        st.write("### Answer")
        st.write(answer)
    else:
        st.write("Please type a question first.")