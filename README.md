# Endee.io AI Assistant: Semantic RAG System

This repository is a forked version of the **Endee Vector Database**, extended with a complete **Retrieval-Augmented Generation (RAG)** pipeline. This project demonstrates how to leverage Endee's high-performance vector indexing to build an intelligent assistant that can chat with technical PDF documentation.

---

## Project Overview
The goal of this project is to provide a seamless interface for querying unstructured technical data. By combining **Endee** as the high-performance storage layer with **Llama 3** as the reasoning engine, the system extracts, indexes, and retrieves precise information from the Endee documentation.

### Key Capabilities:
* **Interactive Web UI**: A Streamlit dashboard for PDF uploading and real-time chatting.
* **CLI Logic Engine**: A terminal-based script for automated RAG flow validation.
* **Smart Context Retrieval**: Uses sliding-window chunking to ensure technical details are preserved.
* **Local-First Privacy**: Entirely local execution using WSL (Endee) and Ollama (LLM).

---

## Knowledge Base
The system is specifically grounded in the **Endee Technical Specification**. 
* **Source File**: `Endee_document.pdf`
* **Content**: This document contains the official architecture, installation guides, API references, and contribution internal protocols for the Endee Vector Database.
* **Grounding**: The RAG pipeline is configured to treat this document as the "Source of Truth," ensuring all AI-generated answers are technically accurate to the Endee ecosystem.

---

## System Design

The architecture is built on a modular RAG pipeline that bridges the gap between raw PDF text and generative AI.

### RAG Pipeline Flow:
`PDF Document` ➔ `PyMuPDF (Extract)` ➔ `SentenceTransformers (Embed)` ➔ **`Endee (Store/Search)`** ➔ `Llama 3 (Generate)` ➔ `User Answer`

### 1. Ingestion Layer (`PyMuPDF`)
Extracts raw text from the source PDF. The text is processed using a **sliding window chunking** strategy (500 characters per chunk with a 100-character overlap). This overlap ensures that semantic meaning and technical context aren't lost at chunk boundaries.

### 2. Embedding Layer (`SentenceTransformers`)
Converts text chunks into 384-dimensional dense vectors using the `all-MiniLM-L6-v2` model. This model is chosen for its industry-leading balance of speed and semantic accuracy.

### 3. Storage & Retrieval Layer (**Endee**)
The core of the system. Endee indexes these vectors using the `cosine` similarity metric. When a user issues a query, Endee performs a K-Nearest Neighbors ($k=4$) search to find the most relevant technical snippets.

### 4. Generation Layer (**Ollama / Llama 3**)
The retrieved context is injected into a specialized prompt. The LLM (Llama 3) analyzes the context to generate a factual, grounded answer, specifically instructed to prioritize technical evidence found in headers and bullet points.

---

## Use of Endee
Endee serves as the high-performance backbone for this project, proving its utility in AI workloads:
* **Index Management**: Handles dynamic creation, indexing, and deletion of the `pdf_index`.
* **Vector Operations**: Executes low-latency insertion and similarity searches of high-dimensional embeddings.
* **Efficiency**: Optimized for modern CPU targets (AVX2/NEON), making it the ideal vector store for local developer environments like WSL.

---

## Setup & Installation

### 1. Prerequisites
* **Endee Server**: Must be running in your WSL/Linux environment on `http://localhost:8080`.
* **Ollama**: Must be running on your host machine (Windows/Mac) on `http://localhost:11434` with the `llama3` model pulled.

### 2. Installation
Clone your personal fork and install the Python dependencies:
```bash
git clone <your-fork-link>
cd endee
pip install -r requirements.txt
```

### 3. Usage
You can interact with the system via two different interfaces:

#### **A. Web Dashboard (Interactive)**
Run this command to open the UI in your browser:
```bash
streamlit run app.py
```
Upload your PDF in the sidebar, click "Process PDF," and start chatting.

#### **B. CLI Engine (Terminal Test)**
Run this command to test the logic directly in your terminal:
```bash
python main.py
```
Automatically indexes the Endee_document.pdf and runs a sample query suite.

---

## Project Structure

- **app.py** – Streamlit frontend and interactive UI logic.  
- **main.py** – Standalone CLI engine for automated validation.  
- **requirements.txt** – Python dependencies.  
- **Endee_document.pdf** – Technical documentation for RAG testing.  
- **ENDEE_ORIGINAL.md** – Original documentation for the Endee project.
