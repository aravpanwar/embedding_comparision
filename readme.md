# LLM Embedding Benchmark: Technical Evaluation

This repository provides a framework to benchmark the performance and efficiency of various Large Language Model (LLM) embedding models. Using an **Apache Spark Analysis Report** as a sample technical dataset, this project evaluates both local and API-based models on their retrieval accuracy (**MRR**, **Recall@3**) and computational latency.

## Benchmark Results

<img width="751" height="992" alt="image" src="https://github.com/user-attachments/assets/08b390cc-33a3-4acf-af25-a6a78002a9e5" />

---

##  Key Technical Features
- **Memory-Safe Benchmarking:** Implements sequential loading and `float16` precision for local models to prevent `OutOfMemory` errors on low-RAM hardware.
- **RAG Evaluation Metrics:** Uses Mean Reciprocal Rank (MRR) and Recall@K to measure retrieval quality.
- **Hybrid Support:** Evaluates both on-premise (Nomic, Qwen, BGE) and cloud-based (Google GenAI, Cohere) embedding providers.
- **Synthetic Q&A Generation:** Includes logic to generate ground-truth testing pairs directly from technical PDF content.

---

# Analysis & Findings

The benchmark reveals that Nomic-V1 is the current "sweet spot" for technical document retrieval on standard hardware. While Qwen-0.6B offers comparable precision, the 12x higher latency on CPU makes it less suitable for real-time applications without GPU acceleration. Surprisingly, local models outperformed general-purpose APIs in Recall@3, suggesting that technical domains benefit significantly from specialized local embedding architectures.
