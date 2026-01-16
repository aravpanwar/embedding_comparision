import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# API SDKs
import google.generativeai as genai
from openai import OpenAI
import cohere

# 1. Load Keys
load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
COHERE_KEY = os.getenv("COHERE_API_KEY")

# 2. Initialize Clients
genai.configure(api_key=GOOGLE_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)
co_client = cohere.ClientV2(COHERE_KEY) # Using V2 Client

# 3. Define Technical Stress Tests (Ubuntu/Linux Focus)
# We test if the model can distinguish between specific technical commands
test_cases = [
    {
        "query": "How to change file permissions in Ubuntu?",
        "match": "You can use the 'chmod' command to modify read/write/execute permissions.",
        "distractor": "The 'chown' command is used to change file ownership, not permissions."
    },
    {
        "query": "Command to list all active network ports?",
        "match": "Use 'sudo netstat -tulpn' or 'ss -tulpn' to see listening ports.",
        "distractor": "The 'ip addr' command shows your network interface IP addresses."
    }
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_benchmark():
    final_data = []

    for test in test_cases:
        q_text = test['query']
        m_text = test['match']
        d_text = test['distractor']

        # --- GOOGLE 005 TEST ---
        try:
            # Note: Google's API maps 004 to 005 automatically now
            q_vec = genai.embed_content(model="models/text-embedding-004", content=q_text, task_type="retrieval_query")['embedding']
            m_vec = genai.embed_content(model="models/text-embedding-004", content=m_text, task_type="retrieval_document")['embedding']
            d_vec = genai.embed_content(model="models/text-embedding-004", content=d_text, task_type="retrieval_document")['embedding']
            
            final_data.append({"Model": "Google-005", "Type": "Correct Match", "Score": cosine_similarity(q_vec, m_vec)})
            final_data.append({"Model": "Google-005", "Type": "Distractor", "Score": cosine_similarity(q_vec, d_vec)})
        except Exception as e: print(f"Google Error: {e}")

        # --- OPENAI SMALL TEST ---
        try:
            q_vec = oa_client.embeddings.create(input=q_text, model="text-embedding-3-small").data[0].embedding
            m_vec = oa_client.embeddings.create(input=m_text, model="text-embedding-3-small").data[0].embedding
            d_vec = oa_client.embeddings.create(input=d_text, model="text-embedding-3-small").data[0].embedding
            
            final_data.append({"Model": "OpenAI-3-Small", "Type": "Correct Match", "Score": cosine_similarity(q_vec, m_vec)})
            final_data.append({"Model": "OpenAI-3-Small", "Type": "Distractor", "Score": cosine_similarity(q_vec, d_vec)})
        except Exception as e: print(f"OpenAI Error: {e}")

        # --- COHERE V3 TEST ---
        try:
            # Cohere requires 'input_type' for their V3 models
            q_vec = co_client.embed(texts=[q_text], model="embed-english-v3.0", input_type="search_query", embedding_types=["float"]).embeddings.float[0]
            m_vec = co_client.embed(texts=[m_text], model="embed-english-v3.0", input_type="search_document", embedding_types=["float"]).embeddings.float[0]
            d_vec = co_client.embed(texts=[d_text], model="embed-english-v3.0", input_type="search_document", embedding_types=["float"]).embeddings.float[0]
            
            final_data.append({"Model": "Cohere-V3", "Type": "Correct Match", "Score": cosine_similarity(q_vec, m_vec)})
            final_data.append({"Model": "Cohere-V3", "Type": "Distractor", "Score": cosine_similarity(q_vec, d_vec)})
        except Exception as e: print(f"Cohere Error: {e}")

    # Process Results
    df = pd.DataFrame(final_data)
    summary = df.groupby(['Model', 'Type'])['Score'].mean().unstack()
    summary['Margin'] = summary['Correct Match'] - summary['Distractor']
    
    print("\n" + "="*50)
    print("EMBEDDING COMPARISON RESULTS")
    print("="*50)
    print(summary.sort_values(by="Margin", ascending=False))
    print("\n*Margin = How much better the model distinguished the right answer.")

if __name__ == "__main__":
    run_benchmark()