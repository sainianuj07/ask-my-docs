# app.py
# A web UI for the RAG app, built with Streamlit.

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import io
import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()

# --- Just set up Gemini and the database ONCE, then reusing them ---
@st.cache_resource
def setup():
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_docs")
    return gemini, collection

gemini, collection = setup()

# --- Reused from load_and_chunk.py / embd_and_store.py ---
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Index an uploaded PDF into the collection ---
def index_pdf(uploaded_file, collection):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(text, chunk_size=400)
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
    new_ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=new_ids)
    return len(chunks)

# --- The answer logic (using same as ask.py, packaged into a function) ---
def answer_question(question):
    results = collection.query(query_texts=[question], n_results=3)
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response.text, chunks

# --- The web page ---
st.title("Ask My Docs")
st.write("Ask a question and get an answer grounded in the documents.")

# --- PDF upload ---
uploaded_file = st.file_uploader("Upload a PDF to query", type="pdf")
if uploaded_file is not None:
    with st.spinner("Indexing PDF..."):
        n = index_pdf(uploaded_file, collection)
    st.success(f"Indexed {n} chunks from {uploaded_file.name}. You can now ask questions about it.")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            answer, chunks = answer_question(question)
        st.write("### Answer")
        st.write(answer)
        st.write("### Sources")
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(f"Source {i}"):
                st.write(chunk)
    else:
        st.write("Please type a question first.")