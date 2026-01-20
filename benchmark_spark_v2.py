import os
import gc
import torch
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
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- 1. MODEL WRAPPERS ---

class NomicWrapper:
    def __init__(self):
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, device='cpu')
    
    def encode(self, texts, convert_to_numpy=True, is_query=False):
        prefix = "search_query: " if is_query else "search_document: "
        prefixed_texts = [prefix + t for t in texts]
        # Nomic uses show_progress_bar internally via SentenceTransformer
        return self.model.encode(prefixed_texts, convert_to_numpy=convert_to_numpy, show_progress_bar=not is_query)

class GoogleWrapper:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    def encode(self, texts, convert_to_numpy=True, is_query=False):
        res = self.client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
            config={"task_type": "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"}
        )
        return np.array([e.values for e in res.embeddings])

class CohereWrapper:
    def __init__(self):
        self.client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
    def encode(self, texts, convert_to_numpy=True, is_query=False):
        res = self.client.embed(texts=texts, model="embed-english-v3.0",
                                input_type="search_query" if is_query else "search_document", 
                                embedding_types=["float"])
        return np.array(res.embeddings.float)

# --- 2. DATA PREPARATION ---

def load_and_chunk_pdf(path):
    print(f"Reading PDF: {path}")
    reader = PdfReader(path)
    text = "".join([page.extract_text() + "\n" for page in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def generate_qa_pairs(chunks):
    qa_pairs = []
    for i, chunk in enumerate(chunks):
        low = chunk.lower()
        q = None
        if "gharchive" in low: q = "What dataset was used for GitHub event data?"
        elif "2,384" in low: q = "How many unique contributors were recorded?"
        elif "11:00 utc" in low: q = "What is the peak daily activity time for Spark development?"
        elif "dongjoon-hyun" in low: q = "Who was the most active contributor identified?"
        elif "tuesday" in low and "thursday" in low: q = "Which days account for the majority of activity?"
        elif "core-periphery" in low: q = "What community structure does the project follow?"
        
        if q and not any(d['question'] == q for d in qa_pairs):
            qa_pairs.append({"question": q, "target_chunk_id": i})
    return qa_pairs

# --- 3. THE BENCHMARK ENGINE ---

def run_benchmark():
    chunks = load_and_chunk_pdf(PDF_PATH)
    dataset = generate_qa_pairs(chunks)
    print(f"Found {len(chunks)} chunks and generated {len(dataset)} questions.\n")

    model_configs = [
        ("BGE-Small (Local)", lambda: SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')),
        ("Nomic-V1 (Local)", lambda: NomicWrapper()),
        ("Google-005 (API)", lambda: GoogleWrapper()),
        ("Cohere-V3 (API)", lambda: CohereWrapper()),
        ("Qwen-0.6B (Local)", lambda: SentenceTransformer(
            'Qwen/Qwen3-Embedding-0.6B', 
            device='cpu', 
            model_kwargs={"dtype": torch.float16}
        ))
    ]
    
    results = []

    for name, loader in model_configs:
        print(f"--- Starting: {name} ---")
        start_time = time.time()
        try:
            if "Qwen" in name:
                print(">>> Initializing Qwen-0.6B (FP16). Please wait...")
            
            model = loader()
            
            # --- FIXED ENCODING LOGIC ---
            print(f"Encoding {len(chunks)} chunks...")
            is_nomic = "Nomic" in name
            is_api = "API" in name

            if is_nomic:
                doc_embs = model.encode(chunks, is_query=False)
            elif is_api:
                # APIs do NOT support show_progress_bar
                doc_embs = model.encode(chunks, convert_to_numpy=True)
            else:
                # Standard Local Models (BGE, Qwen) support progress bars
                doc_embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
            
            doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
            
            ranks = []
            for case in tqdm(dataset, desc=f"Querying {name}"):
                if isinstance(model, (NomicWrapper, GoogleWrapper, CohereWrapper)):
                    q_emb = model.encode([case['question']], is_query=True)[0]
                else:
                    q_emb = model.encode([case['question']], show_progress_bar=False)[0]
                
                q_emb = q_emb / np.linalg.norm(q_emb)
                
                scores = np.dot(doc_embs, q_emb)
                sort_idx = np.argsort(scores)[::-1]
                rank = np.where(sort_idx == case['target_chunk_id'])[0][0] + 1
                ranks.append(rank)
            
            duration = time.time() - start_time
            results.append({
                "Model": name,
                "Recall@3": np.mean([1 if r <= 3 else 0 for r in ranks]),
                "MRR": np.mean([1/r for r in ranks]),
                "Time (s)": round(duration, 2)
            })

        except Exception as e:
            print(f"ERROR testing {name}: {e}")
        finally:
            if 'model' in locals(): del model
            gc.collect()
            print(f"Memory Cleared.\n")

    # Final Output
    df = pd.DataFrame(results).sort_values("MRR", ascending=False)
    print("\n" + "="*60 + "\nAPACHE SPARK REPORT: FINAL BENCHMARK COMPARISON\n" + "="*60)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()