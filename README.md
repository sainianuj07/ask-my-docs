# Ask My Docs — RAG with a Trust Evaluation Layer

A question-answering tool that answers strictly from a set of documents, refuses to guess when the answer isn't there, and automatically grades its own answers for trustworthiness.

## What it does

Index a set of documents, ask questions in plain English, and get answers grounded in those documents. If the answer isn't in the documents, the app replies *"I don't know based on the provided documents"* instead of making something up. A separate evaluation script runs an automated "LLM-as-a-judge" over a test set and produces a trustworthiness scorecard.

## Why I built it

I built this to first understand how RAG actually works in the real world, then how 
applied AI products work end to end — specifically, how to bring product thinking into 
building with RAG. Not just the "AI magic" people talk about every day, but a complete 
pipeline of concrete, debatable decisions. My main goal was to transform myself from 
an AI API automation builder into a builder who understands the tradeoffs that make an 
AI answer trustworthy, and who can measure them.

## How it works (the pipeline)

1. **Load** the documents.
2. **Chunk** them into ~400-character pieces with overlap, so an idea that spans a boundary isn't cut in half.
3. **Embed** each chunk into a vector (a numeric representation of its meaning) using a local embedding model.
4. **Store** the vectors in ChromaDB, a persistent vector database.
5. **Retrieve** — embed the question and pull back the most semantically similar chunks.
6. **Generate** — pass the retrieved chunks plus the question to Gemini, instructed to answer using *only* that context.
7. **Evaluate** — a second model call grades each answer: is every claim supported by the retrieved context?

## Key product decisions

- **Chunk size is a tradeoff.** Too large and retrieval gets vague (a single chunk covers too many ideas); too small and chunks lose the context needed to make sense. I chose 400 characters with overlap.
- **Trust over raw capability.** The model is explicitly instructed to answer only from the documents and to say "I don't know" otherwise. A confident wrong answer is worse than an honest refusal.
- **Evaluation is a feature, not an afterthought.** You can't ship what you can't measure, so the project includes an automated judge that scores trustworthiness (the technical term is *faithfulness*) and flags unsupported answers.
- **Model routing.** Answering uses the faster, cheaper `gemini-2.5-flash-lite`; judging uses the stronger `gemini-2.5-flash` — reserving quality where it matters most and splitting rate-limit quotas across two models.
- **Robust API handling.** Calls retry with a cap on both rate limits (429) and transient server errors (503).
- **Secrets stay out of source control.** The API key lives in a `.env` file, excluded via `.gitignore`.

## Results

On a 5-question test set — 3 answerable from the documents, 2 deliberately not:

| Metric | Result |
|---|---|
| Trustworthiness | 5 / 5 (100%) |
| Untrustworthy answers caught | 0 |

Both out-of-document questions correctly triggered the "I don't know" guardrail rather than answering from the model's general knowledge.

## Tech stack

Python · ChromaDB (vector store + local embeddings) · Google Gemini (`gemini-2.5-flash` / `flash-lite`) · Streamlit (web UI) · python-dotenv

## Limitations & what I'd build next

- **Structured judging.** The judge's PASS/FAIL verdict is parsed from text; a production version would force structured JSON output so the verdict is unambiguous.
- **Bigger, richer evaluation.** Expand the test set and add a retrieval-quality metric (is the *right* chunk retrieved?), not just answer trustworthiness.
- **Smarter chunking.** Try semantic / sentence-aware chunking instead of fixed-size, and measure whether it actually improves answers.
- **Multi-document upload + citations.** Let users upload their own files in the UI and show which chunk each answer came from.

## Running it

1. Create and activate a virtual environment, then install dependencies:
   `pip install chromadb google-genai streamlit pypdf python-dotenv`
2. Add a `.env` file containing `GEMINI_API_KEY=your_key_here`.
3. Build the index once: `python embed_and_store.py`
4. Launch the web app: `streamlit run app.py`
5. Generate the trustworthiness scorecard: `python evaluate.py`
