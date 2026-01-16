import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Modern 2026 Libraries
from google import genai
from openai import OpenAI
import cohere

# 1. Load Keys
load_dotenv()

# 2. Initialize Clients
# Note: google_client now uses the modern 'genai' SDK
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
co_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

# 3. Test Cases (Ubuntu/Linux Focus)
test_cases = [
    {
        "query": "How to change file permissions in Ubuntu?",
        "match": "You can use the 'chmod' command to modify read/write/execute permissions.",
        "distractor": "The 'chown' command is used to change file ownership, not permissions."
    }
]

def cosine_similarity(a, b):
    # Standard math to compare how 'close' two vectors are
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_benchmark():
    final_data = []

    for test in test_cases:
        q, m, d = test['query'], test['match'], test['distractor']

        # --- GOOGLE (Using correct name for AI Studio) ---
        print("Running Google Test...")
        try:
            # Use 'text-embedding-004' for AI Studio; it is the 768-dim standard
            q_v = google_client.models.embed_content(model="text-embedding-004", contents=q).embeddings[0].values
            m_v = google_client.models.embed_content(model="text-embedding-004", contents=m).embeddings[0].values
            d_v = google_client.models.embed_content(model="text-embedding-004", contents=d).embeddings[0].values
            
            final_data.append({"Model": "Google-004/005", "Correct": cosine_similarity(q_v, m_v), "Wrong": cosine_similarity(q_v, d_v)})
        except Exception as e:
            print(f"Google Error: {e}")

        # --- OPENAI (Only runs if quota exists) ---
        print("Running OpenAI Test...")
        try:
            q_oa = oa_client.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
            m_oa = oa_client.embeddings.create(input=m, model="text-embedding-3-small").data[0].embedding
            d_oa = oa_client.embeddings.create(input=d, model="text-embedding-3-small").data[0].embedding
            
            final_data.append({"Model": "OpenAI-3-Small", "Correct": cosine_similarity(q_oa, m_oa), "Wrong": cosine_similarity(q_oa, d_oa)})
        except Exception as e:
            print("OpenAI Skipped: Check billing/quota.")

        # --- COHERE V3 ---
        print("Running Cohere Test...")
        try:
            res = co_client.embed(texts=[q, m, d], model="embed-english-v3.0", input_type="search_query", embedding_types=["float"])
            v = res.embeddings.float
            final_data.append({"Model": "Cohere-V3", "Correct": cosine_similarity(v[0], v[1]), "Wrong": cosine_similarity(v[0], v[2])})
        except Exception as e:
            print(f"Cohere Error: {e}")

    # --- RESULTS ---
    if final_data:
        df = pd.DataFrame(final_data)
        df['Margin'] = df['Correct'] - df['Wrong']
        print("\n" + "="*50)
        print("FINAL COMPARISON")
        print("="*50)
        print(df[['Model', 'Correct', 'Wrong', 'Margin']].sort_values(by="Margin", ascending=False))
    else:
        print("No results collected.")

if __name__ == "__main__":
    run_benchmark()