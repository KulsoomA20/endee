import os
import warnings

#  SET THESE BEFORE ANYTHING ELSE 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

import streamlit as st
import fitz  
import requests
import msgpack
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
BASE_URL = "http://localhost:8080"
OLLAMA_URL = "http://localhost:11434"

# Page Config
st.set_page_config(page_title="Endee PDF Chat", layout="wide")
st.title(" PDF RAG with Endee Vector DB")

# Initialize Model & Mapping
@st.cache_resource
def load_resources():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_resources()

if "id_map" not in st.session_state:
    st.session_state.id_map = {}

# Endee Helper Functions 
def check_services():
    # Verify that both Endee and Ollama are reachable.
    try:
        # Check Endee
        requests.get(f"{BASE_URL}/api/v1/index/default/search", timeout=2)
        # Check Ollama
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False
    
def setup_index():
    requests.delete(f"{BASE_URL}/api/v1/index/pdf_index/delete")
    requests.post(f"{BASE_URL}/api/v1/index/create", json={
        "index_name": "pdf_index", "dim": 384, "space_type": "cosine"
    })

def insert_into_endee(text):
    unique_id = str(uuid.uuid4())
    embedding = model.encode(text).tolist()
    requests.post(f"{BASE_URL}/api/v1/index/pdf_index/vector/insert", 
                  json={"id": unique_id, "vector": embedding})
    st.session_state.id_map[unique_id] = text

# Service Check UI 
if not check_services():
    st.error(" Services Offline! Ensure Endee (WSL) and Ollama (Windows) are running.")
    st.stop()

# UI Sidebar: Upload 
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if st.button("Process PDF"):
        if uploaded_file:
            with st.spinner("Processing..."):
                setup_index()
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                
                full_text = ""
                for page in doc:
                    full_text += page.get_text() + " "
                
                # Smart Chunking: 500 characters per chunk, 100 char overlap
                chunk_size = 500
                overlap = 100
                
                chunks = []
                for i in range(0, len(full_text), chunk_size - overlap):
                    chunk = full_text[i : i + chunk_size]
                    if len(chunk) > 50: # Ignore tiny fragments
                        chunks.append(chunk)
                
                # Insert chunks into Endee
                for c in chunks:
                    insert_into_endee(c)
                    
                st.success(f"Indexed {len(chunks)} technical chunks in Endee!")
        else:
            st.error("Please upload a file first.")

#  Main UI: Chat 
#  Suggested Questions 
suggested_questions = [
    "Type a question...",
    "What language is Endee implemented in?",
    "What is the maximum number of vectors Endee can handle on a single node?",
    "What CPU-targeted optimizations does Endee support?",
    "How does Endee handle metadata-aware retrieval?",
    "What are the primary use cases for Endee?",
    "How do you perform a fast local installation of Endee?",
    "Does Endee support hybrid search?",
    "How can a developer contribute to the project?",
]

selected_suggested = st.selectbox("Suggested Questions:", suggested_questions)

# If user picks a suggestion, it becomes the default value for the text input
default_query = "" if selected_suggested == "Type a question..." else selected_suggested

query = st.text_input("Ask a question about your PDF (or edit the suggestion):", value=default_query)

if st.button("🔍 Search"):
    if query:
        with st.spinner("Thinking..."):
            # 1. Search Endee
            query_vec = model.encode(query).tolist()
            res = requests.post(f"{BASE_URL}/api/v1/index/pdf_index/search", 
                                json={"vector": query_vec, "k": 4})
            
            try:
                decoded = msgpack.unpackb(res.content, raw=False)
                context_list = [st.session_state.id_map.get(r[1], "") for r in decoded]
                context = "\n".join(context_list)
                
                # 2. Generate with Ollama
                prompt = f'''Answer the question using ONLY the context provided. 
                        Rules:
                        - Thorougly go through the file.
                        - Treat headers and bullet points as factual statements of support.
                        - If the context mentions specific numbers (e.g., 8080), languages (e.g., C++), or actions (e.g., Pull Requests), include them in your answer.
                        - If the information is missing, say "Not found in context."\nContext: {context}\nQuestion: {query}'''
                ans_res = requests.post(f"{OLLAMA_URL}/api/generate", 
                                        json={"model": "llama3", "prompt": prompt, "stream": False})
                
                st.markdown("### 🤖 Answer")
                st.write(ans_res.json().get("response"))
                
                with st.expander("See Retrieved Context (from Endee)"):
                    st.write(context_list)
            except:
                st.error("Could not find relevant info or Ollama is offline.")

    else:
        st.warning("Please enter a question or select a suggestion first.")