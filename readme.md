# LLM Embedding Benchmark: Technical Evaluation

This repository provides a framework to benchmark the performance and efficiency of various Large Language Model (LLM) embedding models. Using an **Apache Spark Analysis Report** as a sample technical dataset, this project evaluates both local and API-based models on their retrieval accuracy (**MRR**, **Recall@3**) and computational latency.

## Benchmark Results
============================================================
APACHE SPARK REPORT: FINAL BENCHMARK COMPARISON
============================================================
            Model  Recall@3      MRR  Time (s)
 Nomic-V1 (Local)       0.6 0.624444     26.46
Qwen-0.6B (Local)       0.6 0.622500    320.18
  Cohere-V3 (API)       0.4 0.457524      2.46
BGE-Small (Local)       0.6 0.445000      7.51
 Google-005 (API)       0.4 0.438889      2.24

---

##  Key Technical Features
- **Memory-Safe Benchmarking:** Implements sequential loading and `float16` precision for local models to prevent `OutOfMemory` errors on low-RAM hardware.
- **RAG Evaluation Metrics:** Uses Mean Reciprocal Rank (MRR) and Recall@K to measure retrieval quality.
- **Hybrid Support:** Evaluates both on-premise (Nomic, Qwen, BGE) and cloud-based (Google GenAI, Cohere) embedding providers.
- **Synthetic Q&A Generation:** Includes logic to generate ground-truth testing pairs directly from technical PDF content.

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate

Analysis & Findings

The benchmark reveals that Nomic-V1 is the current "sweet spot" for technical document retrieval on standard hardware. While Qwen-0.6B offers comparable precision, the 12x higher latency on CPU makes it less suitable for real-time applications without GPU acceleration. Surprisingly, local models outperformed general-purpose APIs in Recall@3, suggesting that technical domains benefit significantly from specialized local embedding architectures.