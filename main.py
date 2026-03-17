import os
import warnings

# 1. SET THESE BEFORE ANYTHING ELSE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 2. Suppress the tensorflow warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")

import logging
import fitz  
import requests
import msgpack
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer

# 3. Configuration 
BASE_URL = "http://localhost:8080"
OLLAMA_URL = "http://localhost:11434"
INDEX_NAME = "pdf_index"

# In-memory mapping of Vector IDs to Text
id_map = {}

# 4. Core Functions 
def check_services():
    """Verify that both Endee and Ollama are reachable."""
    try:
        requests.get(f"{BASE_URL}/api/v1/index/default/search", timeout=2)
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        print("All services online.")
        return True
    except Exception:
        print("Services offline. Ensure Endee (WSL) and Ollama (Windows) are running.")
        return False

def setup_index():
    """Reset and initialize the Endee index."""
    requests.delete(f"{BASE_URL}/api/v1/index/{INDEX_NAME}/delete")
    requests.post(f"{BASE_URL}/api/v1/index/create", json={
        "index_name": INDEX_NAME, "dim": 384, "space_type": "cosine"
    })

def load_model():
    """Load the embedding model."""
    print("Loading Embedding Model...")
    return SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf_cli(file_path, model):
    """Extracts, chunks, and indexes PDF text with progress tracking."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False
    
    print(f"Reading {file_path}...")
    doc = fitz.open(file_path)
    full_text = " ".join([page.get_text() for page in doc])
    
    # Sliding window chunking
    chunk_size, overlap = 500, 100
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
    
    total = len(chunks)
    print(f"Indexing {total} chunks into Endee...")
    
    for idx, text in enumerate(chunks, 1):
        if len(text.strip()) > 50:
            # Show progress in terminal
            print(f"\r   → Processing chunk {idx}/{total}...", end="", flush=True)
            
            unique_id = str(uuid.uuid4())
            emb = model.encode(text).tolist()
            requests.post(f"{BASE_URL}/api/v1/index/{INDEX_NAME}/vector/insert", 
                          json={"id": unique_id, "vector": emb})
            id_map[unique_id] = text
            
    print("\nIndexing Complete!")
    return True


def search_and_ask(query, model):
    """Core RAG retrieval and generation logic."""
    query_vec = model.encode(query).tolist()
    res = requests.post(f"{BASE_URL}/api/v1/index/{INDEX_NAME}/search", 
                        json={"vector": query_vec, "k": 3})
    
    try:
        decoded = msgpack.unpackb(res.content, raw=False)
        context_list = [id_map.get(r[1], "") for r in decoded]
        context = "\n".join(context_list)
        
        prompt = f"Using ONLY the context, answer the question.\nContext: {context}\nQuestion: {query}"
        ans_res = requests.post(f"{OLLAMA_URL}/api/generate", 
                                json={"model": "llama3", "prompt": prompt, "stream": False})
        return ans_res.json().get("response", "No response generated.")
    except Exception as e:
        return f"Error during retrieval: {e}"

# 5. Main Flow 
if __name__ == "__main__":
    if not check_services():
        exit()

    setup_index()
    model = load_model()
    
    # document path
    pdf_path = "Endee_document.pdf" 
    
    if process_pdf_cli(pdf_path, model):
        print("\nSystem Ready!")
        
        #  Suggested Questions Menu 
        print("\n--- Suggested Questions ---")
        suggestions = [
            "What language is Endee implemented in?",
            "What is the maximum number of vectors Endee can handle?",
            "What CPU-targeted optimizations does Endee support?",
            "How does Endee handle metadata-aware retrieval?",
            "What are the primary use cases for Endee?"
        ]
        for idx, q in enumerate(suggestions, 1):
            print(f"{idx}. {q}")
        
        print("\nYou can copy-paste these or type your own. Or Type 'exit' to quit.")

        while True:
            user_query = input("\nAsk a question: ")
            
            if user_query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_query.strip():
                continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                answer = search_and_ask(user_query, model)
            
            print(f"\n LLM ANSWER:\n{answer}")
            print("-" * 50)