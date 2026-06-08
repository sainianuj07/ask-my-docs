# embed_and_store.py
# turn chunks into vectors, store them, and test retrieval.

import chromadb

# --- 1. Load and chunk (same as before, now with chunk_size 400) ---
with open("sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(text, chunk_size=400, overlap=50)
print(f"Created {len(chunks)} chunks.")

# --- 2. Open the vector database (a folder on disk) ---
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="my_docs")

# --- 3. Store the chunks. Chroma turns each one into a vector for us. ---
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.upsert(documents=chunks, ids=ids)
print(f"Stored {collection.count()} chunks in the database.")

# --- 4. Test retrieval: ask a question, get the most similar chunks ---
question = "Why does chunk size matter?"
results = collection.query(query_texts=[question], n_results=2)

print(f"\nQuestion: {question}\n")
for i, chunk in enumerate(results["documents"][0]):
    print(f"----- Match {i+1} -----")
    print(chunk)
    print()