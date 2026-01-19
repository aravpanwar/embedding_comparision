import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
import cohere
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
PDF_PATH = "BDA_FINAL_REPORT-2.pdf" 
CHUNK_SIZE = 500
TEST_SIZE = 15  # Number of questions to generate

# --- 1. MODEL WRAPPERS (The Fix for APIs) ---
class GoogleGenAIWrapper:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def encode(self, texts, convert_to_numpy=True):
        # Gemini handles lists, but we process one batch for safety
        try:
            res = self.client.models.embed_content(
                model="text-embedding-004",
                contents=texts
            )
            # Extract embeddings and convert to numpy array
            embeddings = [e.values for e in res.embeddings]
            return np.array(embeddings)
        except Exception as e:
            print(f"Google API Error: {e}")
            return np.zeros((len(texts), 768)) # Return zeros on fail to prevent crash

class CohereWrapper:
    def __init__(self):
        self.client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

    def encode(self, texts, convert_to_numpy=True):
        try:
            # v2 API call
            res = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document", 
                embedding_types=["float"]
            )
            # v2 returns .embeddings.float directly
            return np.array(res.embeddings.float)
        except Exception as e:
            print(f"Cohere API Error: {e}")
            return np.zeros((len(texts), 1024))

# --- 2. PDF PROCESSING ---
def load_and_chunk_pdf(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        print("PDF not found! Please check the filename.")
        return []
    
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Recursive splitting keeps sentences together (Critical for RAG)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"Split PDF into {len(chunks)} chunks.")
    return chunks

# --- 3. GENERATE "GOLDEN DATASET" (Context-Aware) ---
def generate_qa_pairs(chunks):
    qa_pairs = []
    print("Generating Synthetic Questions based on Spark Report...")
    
    # We look for keywords in your PDF to create "Ground Truth" questions
    for i, chunk in enumerate(chunks):
        q = None
        lower_chunk = chunk.lower()
        
        # Logic specific to your BDA Report content
        if "gharchive" in lower_chunk:
            q = "What dataset was used to analyze Spark's history?"
        elif "core-periphery" in lower_chunk:
            q = "What structure does the Spark contributor community follow?"
        elif "churn" in lower_chunk:
            q = "What metric is used to measure code stability and maturity?"
        elif "2,384" in lower_chunk or "unique contributors" in lower_chunk:
            q = "How many unique contributors were found in the study?"
        elif "tuesday" in lower_chunk and "thursday" in lower_chunk:
            q = "On which days is development activity highest?"
        elif "pull request" in lower_chunk and "review" in lower_chunk:
            q = "What activity dominates the Spark development workflow?"
            
        if q:
            # Only add if we haven't asked this specific question recently
            if not any(d['question'] == q for d in qa_pairs):
                qa_pairs.append({
                    "question": q,
                    "target_chunk_id": i,
                    "target_text": chunk
                })
    
    print(f"Generated {len(qa_pairs)} ground-truth test cases.")
    return qa_pairs

# --- 4. SCORING LOGIC ---
def get_rank(query_vec, corpus_vecs, target_id):
    # Dot product similarity
    scores = np.dot(corpus_vecs, query_vec)
    ranked_indices = np.argsort(scores)[::-1]
    # Find rank (1-based index)
    rank = np.where(ranked_indices == target_id)[0][0] + 1
    return rank

# --- 5. MAIN BENCHMARK LOOP ---
def run_benchmark():
    # A. Load Data
    chunks = load_and_chunk_pdf(PDF_PATH)
    if not chunks: return
    
    dataset = generate_qa_pairs(chunks)
    if not dataset:
        print("Could not generate enough questions. Check PDF content.")
        return

    # B. Define Models (Mix of Local + API)
    models = {
        "Qwen-0.6B (Local)": SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cpu'),
        "BGE-Small (Local)": SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'),
        "Google-005 (API)": GoogleGenAIWrapper(),
        "Cohere-V3 (API)": CohereWrapper()
    }
    
    results = []

    # C. Run Tests
    for model_name, model in models.items():
        print(f"\nBenchmarking {model_name}...")
        
        # 1. Embed the "Haystack" (All PDF Chunks)
        # Note: For APIs, this might take a few seconds
        try:
            corpus_embeddings = model.encode(chunks, convert_to_numpy=True)
            
            # Normalize for Cosine Similarity
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            
            recalls = []
            mrrs = []
            
            # 2. Ask Questions
            for case in tqdm(dataset, desc=f"Queries ({model_name})"):
                # Embed query
                q_emb = model.encode([case['question']], convert_to_numpy=True)[0]
                q_emb = q_emb / np.linalg.norm(q_emb)
                
                # Rank
                rank = get_rank(q_emb, corpus_embeddings, case['target_chunk_id'])
                
                # Metrics
                is_hit = 1 if rank <= 3 else 0  # Did it appear in Top 3?
                mrr = 1 / rank                  # Reciprocal Rank
                
                recalls.append(is_hit)
                mrrs.append(mrr)
                
            results.append({
                "Model": model_name,
                "Recall@3": np.mean(recalls),
                "MRR": np.mean(mrrs)
            })
            
        except Exception as e:
            print(f"Failed to benchmark {model_name}: {e}")

    # D. Final Report
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RAG PERFORMANCE: APACHE SPARK REPORT")
    print("="*50)
    print(df.sort_values(by="MRR", ascending=False))

if __name__ == "__main__":
    run_benchmark()
