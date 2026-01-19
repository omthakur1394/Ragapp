# 🧠 DeepSeek "Bionic" RAG Engine

> **"LLMs are blind. I gave mine eyes."**

A production-grade **Retrieval Augmented Generation (RAG)** system engineered to solve the "Table Problem" in scientific papers. Unlike standard RAG implementations that hallucinate on complex layouts, this system uses **Docling** as a visual cortex to correctly interpret mathematical notation (like `-∞` in Softmax) and multi-column tables.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20LangChain%20%7C%20Llama3-blue)

---

## 🚀 The Problem vs. My Solution

| **Standard RAG** ❌ | **Bionic RAG (This Project)** ✅ |
|---------------------|-----------------------------------|
| **Blindsided by Layouts:** Reads PDF text linearly, crushing tables into garbage strings. | **Visual Awareness:** Uses **Docling** to parse document layout, preserving table structures and headers. |
| **Fails Math:** Interprets `-∞` as `?` or `0`, causing model hallucinations. | **Math Precision:** Correctly extracts and embeds mathematical symbols for accurate reasoning. |
| **Amnesia:** Treats every question as a new chat. | **Context-Aware:** Implements **LCEL History Chains** to handle "it," "that," and "the latter" references. |

---

## 🛠️ Tech Stack

* **Ingestion Engine:** `Docling` (Layout-aware PDF parsing)
* **Orchestration:** `LangChain` (Pure **LCEL** Pipelines, no legacy Chains)
* **Vector Database:** `ChromaDB` (Persistent local storage)
* **LLM Inference:** `Llama-3.1-8b` via **Groq** (Sub-second latency)
* **Embeddings:** `HuggingFace` (`all-MiniLM-L6-v2`)
* **Backend:** `FastAPI` with `Lifespan` events & Token Security.

---

## 📂 Project Structure

This project follows a **Modular Microservice** architecture:

```bash
├── 📁 data/               # PDF storage
├── 📁 chroma_db/          # Persistent Vector Database
├── 📄 ingest.py           # The ETL Pipeline (Docling -> Text Splitter -> Chroma)
├── 📄 rag_chain.py        # The Brain (LCEL Logic: Rewrite -> Retrieve -> Answer)
├── 📄 api.py              # The Interface (FastAPI + API Key Security)
├── 📄 vectorstore.py      # Database Connection Logic
├── 📄 config.py           # Configuration Management
└── 📄 .env                # API Keys (Groq, Auth)