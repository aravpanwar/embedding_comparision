import os
import gc
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# SDKs
from sentence_transformers import SentenceTransformer
from google import genai
import cohere

# 1. SETUP
load_dotenv()
query = "How to change file permissions in Ubuntu?"
match = "You can use the 'chmod' command to modify read/write/execute permissions."
distractor = "The 'chown' command is used to change file ownership, not permissions."

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

results = []

# --- 2. API TESTS (No RAM impact) ---
print("--- Starting API Tests ---")

# Google
try:
    google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # In 2026, 'text-embedding-004' is the stable target for AI Studio
    res = google_client.models.embed_content(model="text-embedding-004", contents=[query, match, distractor])
    v = [e.values for e in res.embeddings]
    results.append({"Model": "Google-005", "Correct": cosine_similarity(v[0], v[1]), "Wrong": cosine_similarity(v[0], v[2])})
    print("Google success.")
except Exception as e: print(f"Google failed: {e}")

# Cohere (Updated 2026 SDK Handling)
# API: Cohere (Fixed for 2026 SDK)
try:
    co_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
    res = co_client.embed(
        texts=[query, match, distractor], 
        model="embed-english-v3.0", 
        input_type="search_query", 
        embedding_types=["float"]
    )
    # The fix: Access the .float attribute of the embeddings object
    v = res.embeddings.float 
    
    results.append({
        "Model": "Cohere-V3", 
        "Correct": cosine_similarity(v[0], v[1]), 
        "Wrong": cosine_similarity(v[0], v[2])
    })
    print("Cohere success.")
except Exception as e: 
    print(f"Cohere failed: {e}")


# --- 3. LOCAL TESTS (Sequential Loading for 8GB RAM) ---
print("\n--- Starting Local Tests ---")

# Test BGE-Small
print("Loading BGE-Small...")
try:
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    v = model.encode([query, match, distractor])
    results.append({"Model": "BGE-Small (Local)", "Correct": cosine_similarity(v[0], v[1]), "Wrong": cosine_similarity(v[0], v[2])})
    # Force RAM cleanup
    del model
    gc.collect()
    print("BGE success.")
except Exception as e: print(f"BGE failed: {e}")

# Test Qwen-0.6B (Loaded ONLY after BGE is gone)
print("Loading Qwen-0.6B...")
try:
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cpu')
    v = model.encode([query, match, distractor])
    results.append({"Model": "Qwen-0.6B (Local)", "Correct": cosine_similarity(v[0], v[1]), "Wrong": cosine_similarity(v[0], v[2])})
    del model
    gc.collect()
    print("Qwen success.")
except Exception as e: print(f"Qwen failed: {e}")


# --- 4. FINAL REPORT ---
if results:
    df = pd.DataFrame(results)
    df['Margin'] = df['Correct'] - df['Wrong']
    print("\n" + "="*60)
    print("FINAL CONSOLIDATED BENCHMARK")
    print("="*60)
    print(df[['Model', 'Correct', 'Wrong', 'Margin']].sort_values(by="Margin", ascending=False))
else:
    print("Error: No data points collected.")