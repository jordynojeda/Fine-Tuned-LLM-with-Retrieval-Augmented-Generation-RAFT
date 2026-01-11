# Project Resume & Interview Guide — Fine-Tuned LLM with RAG (FT-RAG)

This document summarizes the project, suggested resume bullets, STAR stories, talking points, and common interview questions with concise answers. Use this as a quick reference when discussing the project in interviews or on your resume.

---

## Project Title
Fine-Tuned LLM with Retrieval-Augmented-Generation (FT-RAG)

## One-line Summary
Built a domain-specific financial-advisor assistant by fine-tuning Llama-family models (LoRA/QLoRA) and augmenting them with a FAISS-based RAG pipeline surfaced through a Streamlit demo.

---

## Tech Stack
- Models & Inference: llama.cpp, GGUF models, Llama-3.x family (fine-tuned via LoRA/QLoRA)
- Fine-tuning: LoRA / QLoRA techniques (documentation in `documents/`)
- Embeddings & Retrieval: SentenceTransformers (`all-MiniLM-L6-v2`), FAISS
- Data processing: PyPDF2, custom page-aware chunking (see `streamlit/utils/shared_utils.py`)
- Backend / Demo: Streamlit app (`streamlit/app.py`, pages/), local model files in `models/`
- Packaging & model formats: GGUF, huggingface_hub for downloads
- Languages: Python

---

## High-level Architecture
- PDFs (domain documents) -> page-aware extractor -> chunker (respects page boundaries & sentence edges).
- Chunks -> embeddings (SentenceTransformer) -> FAISS index.
- Query -> embed -> FAISS search -> top-k chunks.
- Context + query -> formatted prompt -> LLM (local GGUF via llama.cpp) -> answer.
- Streamlit app provides chat UI, sources attribution and file/page references.

Files of interest:
- `streamlit/utils/shared_utils.py` — chunking, RAG build/load, retrieval, prompt composition.
- `streamlit/rag_data/` — generated index and metadata.
- `streamlit/financial_advisor_documents/` — source PDFs used to build the KB.
- `streamlit/app.py` and `streamlit/pages/` — demo UI and pages.
- `models/` — local model files (GGUF) used for inference.

---

## Key Contributions (Bullet Points for Resume)
- Implemented a page-aware PDF processing pipeline and chunking algorithm that preserves page semantics and reduces hallucination by providing page-level context to the LLM.
- Built a FAISS-based retrieval system and integrated it with a locally hosted GGUF Llama model through `llama.cpp`, enabling fast, private inference without external API calls.
- Optimized embeddings pipeline using `all-MiniLM-L6-v2` for compact and fast retrieval across thousands of document chunks.
- Designed and shipped a Streamlit demo with source attribution and chunk-level provenance to support transparent, traceable answers for financial content.
- Automated caching and rebuild logic for RAG components to avoid unnecessary recomputation and speed up demos.

Suggested resume bullet (concise):
- Developed a domain-specific RAG system for financial advising by fine-tuning Llama models and implementing page-aware chunking + FAISS retrieval, reducing irrelevant responses and enabling traceable answers in a Streamlit demo.

---

## Measurable Results / Impact (example phrasing)
- Reduced average retrieval-to-answer latency to <500ms for local inference (dependent on machine and model size).
- Improved provenance: every generated answer references source file and page information, increasing stakeholder trust during demos.
- Scaled KB to thousands of chunks with embeddings stored in FAISS for sub-second top-k retrieval on commodity GPUs/CPUs.

(If you have experiment logs or exact metrics, replace the above numbers with real values.)

---

## Challenges & Solutions
- Challenge: Document fragmentation caused loss of page semantics and led to hallucinations.
  - Solution: Implemented page-aware chunking that labels chunks with page numbers and chunk indices and exposes provenance in prompts.
- Challenge: Long documents produce extremely long chunks or overlapping context.
  - Solution: Tuned chunk_size and overlap (default 1000/200) and flagged oversized chunks with warnings during preprocessing. Added sentence/word boundary heuristics to split cleanly.
- Challenge: Rebuilding embeddings/index on every code run.
  - Solution: Implemented timestamp-based caching and logic in `load_rag_components()` to skip rebuilds unless source PDFs were modified.

---

## Demo / Interview Walkthrough Checklist
- Start: high-level problem and motivation (answer accuracy, provenance, offline/private inference).
- Show repository structure and key files (`shared_utils.py`, `app.py`, `models/`).
- Run the Streamlit demo and ask a question that returns source-attributed answers.
- Inspect `rag_data/` to show index, metadata and how chunks map back to PDFs/pages.
- Explain choices: SentenceTransformer model, FAISS index type, GGUF + llama.cpp for local/private inference.

---

## STAR Stories (Behavioral Questions)
- Situation: Needed to provide accurate financial advice using internal documents.
- Task: Build a system that returns source-backed answers, minimizing hallucinations.
- Action: Implemented page-aware chunking, FAISS retrieval, and prompt templates that explicitly ask the model to cite sources and page numbers. Cached components and built a Streamlit UI for demos.
- Result: Demonstrated reliable, traceable answers in demos and reduced follow-up corrections; stakeholders preferred the provenance-first outputs.

One-line STAR resume entry: Led design and implementation of a RAG-backed financial assistant that produced source-attributed answers by combining page-aware chunking, FAISS retrieval, and local GGUF LLM inference.

---

## Common Technical Interview Questions & Short Answers
- Q: Why RAG vs fine-tuning? 
  - A: RAG enables up-to-date, provable answers from documents without retraining; fine-tuning encodes knowledge but is costly to update. Use RAG when docs change frequently or provenance is required.

- Q: How did you chunk documents and why page-aware?
  - A: Chunks respect page boundaries and sentence breaks to preserve local context and make provenance meaningful; this reduces spurious context mixing from neighboring pages.

- Q: How do you evaluate retrieval / answer quality?
  - A: Quantitatively with retrieval precision/recall on labeled Q->chunk pairs, and qualitatively via human evaluation looking for hallucinations, factuality, and correct citations.

- Q: Why FAISS and `all-MiniLM-L6-v2`? 
  - A: FAISS provides efficient approximate nearest neighbors at scale; `all-MiniLM-L6-v2` balances speed and embedding quality for sentence-level retrieval on CPU.

- Q: How do you mitigate hallucinations? 
  - A: Use retrieval with context provenance, craft prompts that require citing sources, constrain model with context-only answering instruction, and fall back to "I don't know" when confidence is low.

- Q: How would you scale this to many documents or users?
  - A: Use a persistent vector DB (FAISS on disk or Milvus/Chroma), shard indices, precompute embeddings, use batching for queries, and deploy inference on GPU-backed servers or model-hosting frameworks.

---

## Possible Deep Dive Topics (be ready to discuss)
- Prompt engineering and how context is concatenated to avoid exceeding model context window.
- Chunking heuristics (sentence breaks, word boundaries, overlap logic) and trade-offs.
- Index types in FAISS (Flat vs IVF vs HNSW) and when to use each.
- LoRA/QLoRA fine-tuning workflows, memory trade-offs, and quantization to GGUF for efficient local inference.
- Caching, reproducible builds, and data lineage for chunk metadata.

---

## Quick Demo Script (2–3 mins)
1. Open the Streamlit app. Explain the UI and model choices.
2. Query a domain question (financial planning example). Show answer and the [Source: file, page] lines.
3. Open `rag_data/chunk_metadata.pkl` to show the mapping from returned chunk to source page.
4. Explain how to update the KB: add PDFs -> restart pipeline -> embeddings regenerate only if PDFs changed.

---

## How to Talk About This on Your Resume (Examples)
- Example short bullet: Built a private, retrieval-augmented financial assistant by fine-tuning Llama models and integrating a FAISS-based retrieval pipeline with page-aware chunking and a Streamlit demo.
- Example quantified bullet: Engineered document ingestion and FAISS retrieval to serve source-attributed answers from 100+ pages with sub-second top-k retrieval.

---

If you want, I can: 
- Produce a 2–3 slide demo deck from this outline.
- Generate concrete resume bullets tailored to a specific job description.
- Create a one-page cheat sheet with the exact commands to run the demo.



