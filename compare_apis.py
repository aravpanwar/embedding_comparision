import google.generativeai as genai
from openai import OpenAI
import cohere
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
# Replace these with your actual keys
GOOGLE_KEY = "YOUR_GOOGLE_KEY"
OPENAI_KEY = "YOUR_OPENAI_KEY"
COHERE_KEY = "YOUR_COHERE_KEY"

# Initialize Clients
genai.configure(api_key=GOOGLE_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)
co_client = cohere.Client(COHERE_KEY)

# --- THE TEST CASES ---
# A Query and a list of possible matches (1 correct, 2 distractors)
query = "How do I update the kernel on Ubuntu?"
documents = [
    "To update your system, run: sudo apt update && sudo apt upgrade", # Match
    "The kernel is the core of the operating system.",               # Definition (Relatable)
    "I like eating apples in the park."                              # Distractor
]

results = []

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Test Google text-embedding-005 (The direct 004 replacement)
print("Testing Google 005...")
q_005 = genai.embed_content(model="models/text-embedding-004", content=query, task_type="retrieval_query")['embedding'] # Google maps 004 calls to 005 now
for doc in documents:
    d_vec = genai.embed_content(model="models/text-embedding-004", content=doc, task_type="retrieval_document")['embedding']
    results.append({"Model": "Google-005", "Score": cosine_similarity(q_005, d_vec)})

# 2. Test OpenAI text-embedding-3-small
print("Testing OpenAI Small...")
q_oa = oa_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
for doc in documents:
    d_vec = oa_client.embeddings.create(input=doc, model="text-embedding-3-small").data[0].embedding
    results.append({"Model": "OpenAI-3-Small", "Score": cosine_similarity(q_oa, d_vec)})

# 3. Test Cohere Embed v3
print("Testing Cohere v3...")
q_co = co_client.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings[0]
d_co = co_client.embed(texts=documents, model="embed-english-v3.0", input_type="search_document").embeddings
for i, d_vec in enumerate(d_co):
    results.append({"Model": "Cohere-v3", "Score": cosine_similarity(q_co, d_vec)})

# --- REPORT ---
df = pd.DataFrame(results)
print("\n--- Similarity Results (Higher is better) ---")
print(df)
