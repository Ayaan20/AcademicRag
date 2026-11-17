# Academic Paper RAG: Hybrid Retrieval & Cross-Encoder Reranking


An end-to-end, high-precision Retrieval-Augmented Generation (RAG) system specialized for academic literature search and question answering. The pipeline implements **multi-granularity chunking**, a **granularity-aware query router** (rule-based, with an optional trained Mix-of-Granularity MLP router), **three-way hybrid retrieval** (Dense FAISS + BGE-M3 Lexical + BM25) fused via **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder reranking**, **context expansion**, and local open-weight generation via **Ollama**.

It reproduces the ScholarRAG academic-paper RAG pipeline, replaces its hosted LLM API with a locally served open-weight model, and adds the retrieval ablation the original release does not publish.

---

## 📌 System Architecture

```
                                    ┌─────────────────────┐
                                    │     User Query      │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Query Router      │
                                    │ (granularity pick)  │
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

- **Corpus**: 100 arXiv papers parsed with MinerU into structured blocks (text, tables, equations, section headings).
- **Multi-granularity chunking**: each paper is chunked three ways (`sentence`, `paragraph`, `section`) with overlap, giving ~26k chunks (9,354 / 8,749 / 8,226). *Measured sizes are similar across the three levels (median ≈ 120 words), because MinerU blocks are already paragraph-sized; making the section level genuinely coarser is listed under future work.*
- **Query router**: a rule-based router classifies the query (factual, method, experimental, comparison, …) and picks granularities, per-channel RRF weights and preferred sections. An MLP router over frozen `stsb-roberta-large` sentence embeddings (KL-divergence on soft granularity labels) is included (`src/chunking/train_router.py`) and is used by `final_pipeline.py`/`web_app.py` when trained weights are present.
- **Three-way hybrid retrieval** (top-50 per channel):
  - **Dense**: FAISS inner-product index over 1024-d normalised `BAAI/bge-m3` embeddings.
  - **Learned sparse**: BGE-M3 lexical token weights.
  - **BM25**: exact inverted index, protects exact names (models, datasets, authors).
- **Reciprocal Rank Fusion**: `score(d) = Σ_i w_i / (k + rank_i(d))`, `k = 60`, default weights dense 0.4 / sparse 0.3 / BM25 0.3. Reference and acknowledgement chunks (~15% of the index) are filtered out after fusion.
- **Cross-encoder reranking**: `BAAI/bge-reranker-base` (max 512 tokens) rescores the fused candidates jointly with the query; top-10 kept.
- **Context expansion**: the top anchor chunks are widened with their neighbours from the same paper and section (window 2) before generation.
- **Local generation**: answers are generated by an open-weight model served locally by **Ollama** through its OpenAI-compatible API (`src/rag/local_llm.py`; default `qwen2.5:7b-instruct`; any Ollama model works, e.g. `llama3.2:3b`). No hosted LLM API is needed.

---

## 📊 Evaluation

**Benchmark**: 204 held-out questions (`evaluation_set_100papers.json`) over the same 100-paper corpus, labelled with the source paper, question type and difficulty. (A separate 1,935-question set is used for development / router training.)

**Relevance is paper-level**: a question counts as a hit at rank *k* if a chunk from its source paper is in the top-*k* (the labels give the source paper, not a gold passage). This is lenient compared with passage-level recall; see limitations.

### Retrieval ablation (`python run_ablation.py`)

`run_ablation.py` evaluates four configurations on the same 204 held-out questions, changing only the retrieval channels and the reranker (index, router, query expansion and reference filtering are shared):

| Configuration | Flags |
|---|---|
| BM25 only | `--retriever bm25 --no-reranker` |
| Dense only (FAISS + BGE-M3) | `--retriever dense --no-reranker` |
| Hybrid (Dense + Sparse + BM25, RRF) | `--retriever hybrid --no-reranker` |
| Hybrid + cross-encoder reranker | `--retriever hybrid` |

Results are written to `eval_results/ablation_retrieval.{csv,md}`. *Numbers will be added here after the GPU run; the index rebuild needs a GPU (BGE-M3 over ~26k chunks).*

### Full pipeline (hybrid + reranker), earlier run

| Set | Questions | Hit@10 | MRR@10 | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|---|
| Dev (train) | 1,935 | 0.965 | 0.920 | 0.893 | 0.944 | 0.957 |
| Held-out (eval) | 204 | 0.931 | 0.871 | 0.828 | 0.912 | 0.931 |

LLM-judged answer quality on the held-out set (1–5 scale): faithfulness 4.25, context precision 4.11, answer relevancy 3.95, context recall 3.40; ROUGE-L 0.194, token-F1 0.314 (see `eval_results/`).

### Limitations

- Paper-level relevance overstates passage-level retrieval quality.
- Train and held-out questions are about the same 100 papers (held-out *questions*, not held-out papers).
- The three chunk granularities currently have similar sizes.
- Answer-quality scores use an LLM judge (`llm_evaluator.py`), which has its own biases.

---

## ▶️ Running

```bash
pip install -r requirements.txt
ollama serve & ollama pull qwen2.5:7b-instruct      # local generator
python rebuild_index.py                              # needs parsed_pdf/*_content_list_v2.json
python final_pipeline.py                             # interactive QA in the terminal
python web_app.py                                    # web UI
python run_ablation.py --dataset evaluation_set_100papers.json
```

`LLM_BASE_URL` / `LLM_MODEL` switch the local model. `notebooks/colab_ablation.ipynb` runs index build + ablation + local generation on a Colab T4.

---

## 📁 Repository Structure

```
.
├── src/
│   ├── chunking/     # MinerU → unified format, structure recognition, multi-granularity chunker,
│   │                 #   rule-based query router, MLP (MoG) router + training
│   ├── embedding/    # BGE-M3 dense + sparse embedding, FAISS index builder
│   ├── retrieval/    # dense, BM25, BGE-M3 sparse, hybrid RRF, cross-encoder reranker, context expander
│   ├── rag/          # prompt templates, answer generation, local LLM client (Ollama)
│   └── utils/        # Hugging Face model resolution
├── notebooks/colab_ablation.ipynb
├── rebuild_index.py   # chunk + embed + index the corpus
├── final_pipeline.py  # terminal QA
├── web_app.py         # Flask UI (templates/)
├── batch_evaluate.py  # Hit/MRR/Recall@K (+ ROUGE-L / token-F1 with generation); --retriever bm25|dense|hybrid
├── run_ablation.py    # 4-configuration retrieval ablation
├── llm_evaluator.py   # LLM-as-judge answer quality
└── eval_results/
```

---

## Future work

- Passage-level gold labels for strict retrieval recall.
- Make section chunks genuinely coarser and add a document-level index.
- Benchmark the trained MoG router against the rule-based router.
