import os
import gc
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
import cohere

load_dotenv()

# --- CONFIG ---
query = "How to change file permissions in Ubuntu?"
match = "You can use the 'chmod' command to modify read/write/execute permissions."
distractor = "The 'chown' command is used to change file ownership, not permissions."

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

results = []

# --- 1. API TESTS (Zero RAM Usage) ---
print("Running API tests...")
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
try:
    q_v = google_client.models.embed_content(model="text-embedding-004", contents=query).embeddings[0].values
    m_v = google_client.models.embed_content(model="text-embedding-004", contents=match).embeddings[0].values
    d_v = google_client.models.embed_content(model="text-embedding-004", contents=distractor).embeddings[0].values
    results.append({"Model": "Google-005", "Correct": cosine_similarity(q_v, m_v), "Wrong": cosine_similarity(q_v, d_v)})
except Exception as e: print(f"Google Error: {e}")

# --- 2. LOCAL TEST: BGE-SMALL (Very Low RAM) ---
print("\nLoading BGE-Small...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
q_v, m_v, d_v = model.encode([query, match, distractor])
results.append({"Model": "BGE-Small (Local)", "Correct": cosine_similarity(q_v, m_v), "Wrong": cosine_similarity(q_v, d_v)})

# CLEANUP RAM
del model
gc.collect() 

# --- 3. LOCAL TEST: QWEN-0.6B (Medium RAM) ---
print("Loading Qwen-0.6B...")
# We load this last as it is the heaviest
try:
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cpu')
    q_v, m_v, d_v = model.encode([query, match, distractor])
    results.append({"Model": "Qwen-0.6B (Local)", "Correct": cosine_similarity(q_v, m_v), "Wrong": cosine_similarity(q_v, d_v)})
    del model
    gc.collect()
except Exception as e:
    print(f"Skipping Qwen: Likely ran out of RAM. Error: {e}")

# --- FINAL OUTPUT ---
df = pd.DataFrame(results)
df['Margin'] = df['Correct'] - df['Wrong']
print("\n" + "="*50)
print("FINAL STABLE COMPARISON")
print("="*50)
print(df[['Model', 'Correct', 'Wrong', 'Margin']].sort_values(by="Margin", ascending=False))