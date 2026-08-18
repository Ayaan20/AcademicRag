# Academic Paper RAG: Hybrid Retrieval & Cross-Encoder Reranking

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/FAISS-Dense%20Search-green)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-precision Retrieval-Augmented Generation (RAG) system specialized for academic literature search and question answering. The pipeline implements **multi-granularity chunking**, a **Mix-of-Granularity (MoG) router**, **three-way hybrid retrieval** (Dense FAISS + BGE-M3 Lexical + BM25) fused via **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder reranking**, **hierarchical context expansion**, and local open-weight generation via **Ollama**.

---

## 📌 System Architecture

```
                                    ┌─────────────────────┐
                                    │     User Query      │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │     MoG Router      │
                                    │ (stsb-roberta-large)│
                                    └──────────┬──────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
    ┌──────────▼──────────┐         ┌──────────▼──────────┐         ┌──────────▼──────────┐
    │    Dense FAISS      │         │    BGE-M3 Lexical   │         │     BM25 Sparse     │
    │  (Semantic Vectors) │         │   (Sparse Weights)  │         │   (Lexical Tokens)  │
    └──────────┬──────────┘         └──────────┬──────────┘         └──────────┬──────────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │ Reciprocal Rank     │
                                    │   Fusion (RRF)      │
                                    └──────────┬──────────┘
                                               │ Top-K Candidates
                                    ┌──────────▼──────────┐
                                    │    Cross-Encoder    │
                                    │   (bge-reranker)    │
                                    └──────────┬──────────┘
                                               │ Top-N Chunks
                                    ┌──────────▼──────────┐
                                    │ Context Expansion   │
                                    │ (Sibling Chunks)    │
                                    └──────────┬──────────┘
                                               │ Expanded Context
                                    ┌──────────▼──────────┐
                                    │  Locally Served LLM │
                                    │   (Ollama Engine)   │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Grounded Answer   │
                                    └─────────────────────┘
```

---

## 🚀 Key Features

- **Multi-Granularity Document Representation**: Academic papers are decomposed across 4 hierarchical granularities:
  - **Sentence level** (~100 tokens): Pinpoints exact formulas, hyperparameter settings, and factual data.
  - **Paragraph level** (~300–500 tokens): Captures methodological workflows and experimental setups.
  - **Section level** (~1000–2000 tokens): Preserves end-to-end domain arguments and comparative discussions.
  - **Document level** (~4000+ tokens): High-level abstract summaries and overarching contributions.
- **Mix-of-Granularity (MoG) Router**: A classifier backed by `stsb-roberta-large` that analyzes query intent and predicts the optimal chunk-granularity distribution dynamically.
- **Three-Way Hybrid Retrieval**:
  - **Dense Retrieval**: FAISS index built over 1024-d dense representations from `BAAI/bge-m3`.
  - **Learned Sparse Retrieval**: BGE-M3 lexical token weights capturing domain-specific terminology.
  - **Exact BM25**: Traditional inverted index to prevent semantic drift on precise author/dataset/algorithm names.
- **Reciprocal Rank Fusion (RRF)**: Combines ranked lists from distinct retrieval channels into a unified candidate pool.
- **Cross-Encoder Reranking**: Applies `BAAI/bge-reranker-base` to jointly score query-chunk pairs, eliminating false positives from initial vector search.
- **Hierarchical Context Expansion**: Dynamically expands high-scoring chunks with neighboring sibling chunks from the original paper structure.
- **Locally Served Open-Weight Generation**: Integrated with local LLMs (DeepSeek / Llama 3.2 / Qwen 2.5) via **Ollama**, ensuring complete data privacy with no external API dependencies.

---

## 📊 Retrieval Ablation & Evaluation Results

Comprehensive ablation experiments were executed across a held-out benchmark of **100+ annotated research papers** and question-answer pairs:

| Retrieval Configuration | Recall@5 | Recall@10 | MRR@10 | Context Precision |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Only** | 0.584 | 0.672 | 0.512 | 0.618 |
| **Dense Only (FAISS + BGE-M3)** | 0.712 | 0.804 | 0.658 | 0.742 |
| **Hybrid (Dense + Sparse + BM25 via RRF)** | 0.826 | 0.891 | 0.764 | 0.819 |
| **Hybrid + Cross-Encoder Reranker (Full Pipeline)** | **0.897** | **0.948** | **0.852** | **0.912** |

---

## 📁 Repository Structure

```
.
├── src/
│   ├── chunking/          # Multi-granularity document parser & chunker
│   ├── router/            # MoG query classifier (stsb-roberta-large)
│   ├── retrieval/         # FAISS dense index, BM25, and BGE-M3 sparse retriever
│   ├── reranker/          # Cross-encoder scoring and rank fusion
│   ├── generator/         # Ollama / local LLM streaming client
│   └── pipeline.py        # End-to-end unified orchestration pipeline
├── eval_results/          # Retrieval ablation metrics, logs, and benchmark tables
├── templates/             # Interactive web application templates
├── final_pipeline.py      # Standalone terminal & evaluation driver
├── batch_evaluate.py      # Automated Recall@K and MRR evaluation suite
├── web_app.py             # Interactive research UI
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-username>/ScholarRAG.git
cd ScholarRAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Pretrained Models & Setup Ollama

```bash
# Pull local LLM
ollama pull llama3.2:3b

# Download embedding and reranking models
python download_models.py
```

### 3. Build Vector Indices

```bash
python rebuild_index.py --data_dir ./parsed_pdf
```

### 4. Run Retrieval & Generation

```bash
# Terminal Query
python final_pipeline.py --query "What are the key limitations of LoRA fine-tuning mentioned in the paper?"

# Launch Web UI
python web_app.py
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
