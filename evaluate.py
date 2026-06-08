# evaluate.py
# An evaluation system: automatically grade the RAG app's answers for trustworthiness.

import os
import time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="my_docs")

# --- A safe wrapper that waits out the rate limit, retries for gemini's server overload error and retries ---
def safe_generate(prompt, model, max_retries=6):
    for attempt in range(max_retries):
        try:
            response = gemini.models.generate_content(model=model, contents=prompt)
            return response.text
        except errors.APIError as e:
            message = str(e)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                print("    Rate limit reached — waiting 60s, then retrying...")
                time.sleep(60)
            elif "UNAVAILABLE" in message or "503" in message:
                print("    Model busy (503) — waiting 15s, then retrying...")
                time.sleep(15)
            else:
                raise
    raise RuntimeError("Gave up after several retries — the API kept failing.")

# --- The RAG answer logic, now using the safe wrapper ---
def get_answer_and_context(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    answer = safe_generate(prompt, "gemini-2.5-flash-lite")
    return answer, context

# --- The judge: a SECOND model call that grades trustworthiness ---
def judge_trustworthiness(question, context, answer):
    judge_prompt = f"""You are a strict evaluator checking an AI answer for trustworthiness.
An answer is TRUSTWORTHY only if every claim in it is supported by the Context.
Saying "I don't know based on the provided documents" counts as TRUSTWORTHY.
Using outside knowledge not found in the Context counts as UNTRUSTWORTHY.

Context:
{context}

Question: {question}

Answer to grade:
{answer}

Reply with the single word PASS (trustworthy) or FAIL (untrustworthy) on the first line, then a one-sentence reason."""
    verdict = safe_generate(judge_prompt, "gemini-2.5-flash").strip()
    is_trustworthy = verdict.upper().startswith("PASS")
    return is_trustworthy, verdict

# --- The test set: a mix of in-document and out-of-document questions ---
test_questions = [
    "Why does chunk size matter?",
    "What is RAG?",
    "How are document chunks stored so they can be searched?",
    "What is the capital of France?",
    "Who won the 2019 Cricket World Cup?",
]

# --- Run the evaluation ---
passed = 0
print("Running evaluation...\n")
for i, question in enumerate(test_questions):
    answer, context = get_answer_and_context(question)
    is_trustworthy, verdict = judge_trustworthiness(question, context, answer)
    if is_trustworthy:
        passed += 1
    status = "PASS" if is_trustworthy else "FAIL"
    print(f"[{status}] Q{i+1}: {question}")
    print(f"    Answer: {answer}")
    print(f"    Judge:  {verdict}")
    print()

# --- The scorecard ---
total = len(test_questions)
trustworthiness_rate = passed / total * 100
print("=" * 50)
print("SCORECARD")
print(f"Trustworthiness: {passed}/{total} answers passed ({trustworthiness_rate:.0f}%)")
print(f"Untrustworthy answers caught: {total - passed}")