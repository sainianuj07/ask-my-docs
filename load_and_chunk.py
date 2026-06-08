# load_and_chunk.py
# Step 1 of the RAG pipeline: read a document and split it into chunks.

# --- 1. Read the document from disk ---
with open("sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(f"Loaded the document. It has {len(text)} characters.\n")

# --- 2. Define how we chunk ---
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- 3. Run the chunker ---
chunks = chunk_text(text, chunk_size=400, overlap=50)

print(f"Split the document into {len(chunks)} chunks.\n")

# --- 4. Show the chunks so we can see what happened ---
for i, chunk in enumerate(chunks):
    print(f"----- Chunk {i+1} ({len(chunk)} characters) -----")
    print(chunk)
    print()