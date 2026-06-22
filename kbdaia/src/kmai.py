# version 2.0.5e - 
# calling manage_cache_limit to manage records in SQLite qa_cache table
## - updated manage_cache_limit limit=50
#---------------------------------------------------------------
# Used base as production ready version 2.0.4 and added chromadb, sqlite & restore classes

import os
import io
import time
import sqlite3
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
import numpy as np
import re
import gc

import streamlit as st
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Vector DB (Chroma)
import chromadb
from chromadb import PersistentClient

# PDF handling
import fitz  # PyMuPDF
from pypdf import PdfReader

# OCR for images in PDF (optional)
from pdf2image import convert_from_bytes
import pytesseract

# Table extraction
import camelot  # pip install "camelot-py[cv]"

# parallel summarization
from concurrent.futures import ThreadPoolExecutor, as_completed

# Rough metrics
from rouge_score import rouge_scorer

# NLTK for Adaptive Chunking
import nltk

# Ensure NLTK data is available (runs efficiently on CPU)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# import backup and restore run once utilities
from persistence.restore_once import (
    restore_chroma_once,
    restore_sqlite_once
)

restore_chroma_once()
restore_sqlite_once()

# import backup and restore utilities
from persistence.chroma_backup import backup_chroma
from persistence.sqlite_backup import backup_sqlite

# cache management utility
from persistence.sqlite_mgmt import manage_cache_limit

# Inject custom CSS to hide Streamlit kebab menu i.e. three vertical dots on left
# Manage header font size

combined_style = """
    <style>
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;} /* Optional: Hides the top header bar entirely */

    /* Custom Header Font Sizes */
    .header1 {
        font-size: 20px !important;
        font-weight: bold;
    }
    .header2 {
        font-size: 10px !important;
    }
    </style>
    """
st.markdown(combined_style, unsafe_allow_html=True)

# =============================================
# CONFIG
# =============================================
st.set_page_config(page_title="Enterprise PDF → Vector DB (with Cache)", layout="wide")
st.title("📚 DocIQ - Query your knowledge base")

# gate
ADMIN_USER = "shikari" #"admin"
ADMIN_PASS = "shambu1983" #"secret123"

# New Viewer Credentials
VIEWER_USER = "viewer"
VIEWER_PASS = "view123"

DATA_DIR = "data_enterprise"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CACHE_DB_PATH = os.path.join(DATA_DIR, "cache_store.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

torch.cuda.empty_cache()
gc.collect()

# =============================================
# helper: derive subtopic from filename
# =============================================
def derive_subtopic_from_filename(filename: str) -> str:
    if not filename:
        return "general"
    name = os.path.splitext(filename)[0].replace(" ", "_").strip()
    name = re.sub(r'[-_](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\d{2,4}$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_]\d{4,8}$', '', name)
    parts = [p for p in re.split(r'[_\-]+', name) if p.strip()]
    if len(parts) >= 3:
        candidate = "_".join(parts[-3:])
    elif parts:
        candidate = "_".join(parts[-2:])
    else:
        candidate = "general"
    candidate = re.sub(r'[^A-Za-z0-9_]+', '', candidate).strip("_").lower()
    return candidate if candidate else "general"

# =============================================
# SQLITE CACHE DB (with source_docs, relevance_threshold, top_k)
# =============================================
def init_cache_db(path: str = CACHE_DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    # Create table if not exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qa_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        sub_topic TEXT DEFAULT 'all',
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT NOT NULL,
        feedback_status TEXT CHECK(feedback_status IN ('Y','N')) DEFAULT NULL,
        cosine_score REAL,
        rouge2_score REAL,
        rougeL_score REAL,
        source_docs TEXT,
        relevance_threshold REAL,
        top_k INTEGER
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON qa_cache(topic);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sub_topic ON qa_cache(sub_topic);")
    conn.commit()

    # Migration: ensure columns exist (for older DBs)
    cur.execute("PRAGMA table_info(qa_cache);")
    existing_cols = [r[1] for r in cur.fetchall()]
    # ensure metric/settings columns exist
    expected_cols = {
        "cosine_score": "REAL",
        "rouge2_score": "REAL",
        "rougeL_score": "REAL",
        "source_docs": "TEXT",
        "relevance_threshold": "REAL",
        "top_k": "INTEGER"
    }
    for col, col_type in expected_cols.items():
        if col not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE qa_cache ADD COLUMN {col} {col_type};")
                conn.commit()
            except Exception as e:
                # Ignore failures (older SQLite versions, etc.)
                st.warning(f"Could not add column {col}: {e}")
                pass
    return conn

_cache_conn = init_cache_db()

def fetch_cache_by_topic(topic: str, only_helpful: bool = False):
    """
    Returns rows with columns:
    id, question, answer, created_at, feedback_status, sub_topic, cosine_score, rouge2_score, rougeL_score, source_docs, relevance_threshold, top_k
    """      
    cur = _cache_conn.cursor()
    # We explicitly select columns to ensure order
    cols = "id, question, answer, created_at, feedback_status, sub_topic, cosine_score, rouge2_score, rougeL_score, source_docs, relevance_threshold, top_k"
    
    if only_helpful:
        cur.execute(
            f"SELECT {cols} FROM qa_cache WHERE topic = ? AND feedback_status = 'Y' ORDER BY created_at DESC",
            (topic,)
        )
    else:
        cur.execute(
            f"SELECT {cols} FROM qa_cache WHERE topic = ? ORDER BY created_at DESC",
            (topic,)
        )
    rows = cur.fetchall()
    # convert source_docs JSON text to Python list where possible
    processed = []
    for r in rows:
        r = list(r)
        sd_idx = 9  # index of source_docs column in our SELECT
        try:
            sd_val = r[sd_idx]
            if sd_val:
                r[sd_idx] = json.loads(sd_val)
            else:
                r[sd_idx] = []
        except Exception:
            r[sd_idx] = []
        processed.append(tuple(r))
    return processed

def get_best_fuzzy_match(topic: str, question: str, threshold: float = 0.75):
    """
    Return (id, stored_question, stored_answer, created_at, feedback_status, sub_topic, cosine_score, rouge2_score, rougeL_score, score, source_docs, relevance_threshold, top_k)
    or None if no match above threshold.
    IMPORTANT: Only consider rows with feedback_status='Y' (helpful).
    """                                                                                                                                   
    rows = fetch_cache_by_topic(topic, only_helpful=True)
    if not rows:
        return None

    q_lower = question.lower().strip()
    best = None
    best_score = 0.0
    for row in rows:
        # row = (id, question, answer, created_at, fb, sub_topic, cos_m, r2_m, rL_m, source_docs, rel_th, k)
        rid, stored_q, stored_a, created_at, fb, sub_topic, cos_m, r2_m, rL_m, source_docs, rel_th, k = row
        score = SequenceMatcher(None, q_lower, (stored_q or "").lower().strip()).ratio()
        if score > best_score:
            best_score = score
            best = (rid, stored_q, stored_a, created_at, fb, sub_topic, cos_m, r2_m, rL_m, score, source_docs, rel_th, k)

    if best and best[9] >= threshold:
        return best
    return None

def insert_cache(topic: str, question: str, answer: str, feedback_status: str = None, sub_topic: str = "all",
                 cosine_score: float = None, rouge2_score: float = None, rougeL_score: float = None,
                 source_docs: List[str] = None, relevance_threshold: float = None, top_k: int = None):
    now = datetime.now(timezone.utc).isoformat()
    cur = _cache_conn.cursor()
    sd_text = json.dumps(source_docs or [])
    cur.execute("""
        INSERT INTO qa_cache (topic, sub_topic, question, answer, created_at, feedback_status, cosine_score, rouge2_score, rougeL_score, source_docs, relevance_threshold, top_k)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (topic, sub_topic, question, answer, now, feedback_status, cosine_score, rouge2_score, rougeL_score, sd_text, relevance_threshold, top_k))
    _cache_conn.commit()
    
    # Call the maintenance logic from persistence/sqlite_mgmt.py 
    manage_cache_limit(CACHE_DB_PATH, limit=50) # pass CACHE_DB_PATH and the limit of 100
    backup_sqlite() # backup inserts
    return cur.lastrowid

def upsert_cache(topic: str, question: str, answer: str, feedback_status: str = None, sub_topic: str = "all",
                 cosine_score: float = None, rouge2_score: float = None, rougeL_score: float = None,
                 source_docs: List[str] = None, relevance_threshold: float = None, top_k: int = None):
    """
    Upsert identified by topic, question, AND sub_topic.
    """                                                                                                              
    now = datetime.now(timezone.utc).isoformat()
    sd_text = json.dumps(source_docs or [])
    cur = _cache_conn.cursor()
    cur.execute("SELECT id FROM qa_cache WHERE topic = ? AND question = ? AND sub_topic = ?", (topic, question, sub_topic))
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE qa_cache SET answer = ?, created_at = ?, feedback_status = ?, sub_topic = ?, cosine_score = ?, rouge2_score = ?, rougeL_score = ?, source_docs = ?, relevance_threshold = ?, top_k = ? WHERE id = ?",
            (answer, now, feedback_status, sub_topic, cosine_score, rouge2_score, rougeL_score, sd_text, relevance_threshold, top_k, row[0])
        )
        _cache_conn.commit()
        backup_sqlite() # backup updates
        return row[0]
    else:
        return insert_cache(topic, question, answer, feedback_status, sub_topic, cosine_score, rouge2_score, rougeL_score, source_docs, relevance_threshold, top_k)

def update_feedback(entry_id: int, feedback_status: str):
    try:
        with sqlite3.connect(CACHE_DB_PATH, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE qa_cache SET feedback_status = ? WHERE id = ?", (feedback_status, entry_id))
            conn.commit()
            affected = cur.rowcount
        _cache_conn.commit()
        backup_sqlite() #backup feedback_status updates
        if affected > 0:
            st.toast(f"✓ Feedback updated (id={entry_id}, status={feedback_status})")
        else:
            st.warning(f"[Info] No row found for id={entry_id}.")
    except Exception as e:
        st.error(f"[Debug] Failed to update feedback for id={entry_id}: {e}")

def delete_cache_entry(entry_id: int):
    cur = _cache_conn.cursor()
    cur.execute("DELETE FROM qa_cache WHERE id = ?", (entry_id,))
    _cache_conn.commit()
    backup_sqlite() #back up deletes

def clear_cache_db():
    cur = _cache_conn.cursor()
    cur.execute("DELETE FROM qa_cache;")
    _cache_conn.commit()
    backup_sqlite() #backup sqlite changes

# =============================================
# EMBEDDING MODEL (Sentence-Transformers)
# =============================================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        gc.collect()
        torch.cuda.empty_cache()
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        _ = _embed_model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)
    return _embed_model

def embed_texts(texts: List[str]):
    if not texts:
        return np.array([])
    model = get_embed_model()
    with torch.no_grad():
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# =============================================
# CHROMA CLIENT
# =============================================
chroma_client = PersistentClient(path=CHROMA_DIR)
# =============================================
# VECTOR DB HELPERS
# =============================================                  
def get_or_create_collection(topic_name: str):
    try:
        return chroma_client.get_collection(name=topic_name)
    except Exception:
        return chroma_client.create_collection(name=topic_name, embedding_function=None)

# =============================================
# PDF / OCR / TABLE extraction functions
# =============================================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([p.get_text("text") for p in doc])
    except Exception:
        pass
    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            text = ""
    return text

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes)
    texts = [pytesseract.image_to_string(p) for p in pages]
    return "\n".join(texts)

def extract_tables_as_text(pdf_path: str) -> str:
    table_texts = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for t in tables:
            table_texts.append(t.df.to_string(index=False))
    except Exception:
        pass
    return "\n\n".join(table_texts)

# =============================================
# chunking strategies
# =============================================
def chunk_text_fixed(text: str, chunk_size=1000, overlap=200) -> List[str]:
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")
    chunks, start, L = [], 0, len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
    
def chunk_text_recursive(text: str, chunk_size=1500, overlap=200) -> List[str]:
    """
    Recursive sentence-aware chunking (Regex based):
    """
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    # Split into sentences using Regex lookbehind
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if s.strip()
    ]

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        s_len = len(sent)

        # If adding sentence exceeds chunk size → finalize chunk
        if current and (current_len + s_len + 1 > chunk_size):
            chunk = " ".join(current).strip()
            chunks.append(chunk)

            # Compute overlap using backward sentences
            overlap_sentences = []
            ov_len = 0
            for ss in reversed(current):
                overlap_sentences.insert(0, ss)
                ov_len += len(ss) + 1
                if ov_len >= overlap:
                    break

            current = overlap_sentences.copy()
            current_len = sum(len(x) + 1 for x in current)

        # Add sentence
        current.append(sent)
        current_len += s_len + 1

    # Append remainder
    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks

def chunk_text_adaptive_nltk(text: str, chunk_size=1200, overlap=200) -> List[str]:
    """
    Adaptive Chunking (NLTK): 
    Uses NLTK's robust sentence tokenizer to split text, then groups sentences
    into chunks respecting the token limit. Extremely efficient for 2 vCPUs.
    """
    if not text:
        return []
    
    # NLTK sentence splitting is more accurate than regex for edge cases (Dr., Mr., etc.)
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback if nltk fails
        return chunk_text_recursive(text, chunk_size, overlap)

    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        sent_len = len(sent)
        
        # Check if adding this sentence exceeds chunk_size
        if current_len + sent_len + 1 > chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Calculate overlap for the NEXT chunk
            # We want to keep the last few sentences from the previous chunk 
            # such that their length is roughly ~overlap
            overlap_buffer = []
            overlap_len = 0
            
            for prev_sent in reversed(current_chunk):
                if overlap_len + len(prev_sent) + 1 <= overlap:
                    overlap_buffer.insert(0, prev_sent)
                    overlap_len += len(prev_sent) + 1
                else:
                    break
            
            # Reset current chunk with overlap + new sentence
            current_chunk = overlap_buffer + [sent]
            current_len = overlap_len + sent_len + 1
        else:
            current_chunk.append(sent)
            current_len += sent_len + 1
            
    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def chunk_text(text: str, chunk_size=1000, overlap=200, strategy: str = "fixed") -> List[str]:
    """
    General chunk_text wrapper supporting strategies:
       - strategy='fixed'      : original fixed sliding-window
       - strategy='recursive'  : regex sentence-aware
       - strategy='adaptive'   : NLTK sentence-aware (New)
    """
    # normalize strategy                    
    strategy = (strategy or "fixed").lower()
    try:
        if strategy == "adaptive":
            return chunk_text_adaptive_nltk(text, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "recursive":
            return chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "fixed":
            return chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap)
        else:
            return chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap)
    except Exception as e:
        st.warning(f"Chunking error using '{strategy}', fallback to recursive. Error: {e}")
        try:
            return chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap)
        except Exception:
            return [text] # final safety fallback

# =============================================
# GENERATOR MODEL (Instruct Tuned - LaMini)
# =============================================
# CHANGED FROM SUMMARIZER to QA MODEL
GENERATOR_MODEL_NAME = "MBZUAI/LaMini-Flan-T5-248M"

@st.cache_resource(show_spinner="Loading QA model...")
def load_generator():
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    # Changed pipeline from 'summarization' to 'text2text-generation'
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

generator = load_generator()

def generate_answer_from_context(
    combined_context: str,
    question: str,
    support_threshold: float = 0.60,
    top_k: int = 4
) -> (str, float, float, float):
    """
    New function replacing chunked_summarize.
	Instead of summarizing chunks individually, this takes the combined top-k retrieval 
    and asks the model to answer the question based on that context.														
    """
    start_time = time.perf_counter()
    
    # 1. Truncate context to safe limit for LaMini/T5 (approx 2000-2500 chars)
    # This prevents the model from crashing on 2vCPU
    max_char_limit = 3000
    safe_context = combined_context[:max_char_limit]

    # 2. Construct Prompt
    # Prompt engineering to prevent hallucinations
    input_prompt = f"""Identify the answer to the following question using only the context provided.
If the answer is not found in the context, respond with "Not stated in context".

Context:
{safe_context}

Question: 
{question}

Answer:"""

    # 3. Generate
    try:
        output = generator(
            input_prompt,
            max_length=512,
            do_sample=False,
            temperature=0.0,
            truncation=True
        )
        final_answer = output[0]['generated_text'].strip()
    except Exception as e:
        final_answer = f"[Generation failed: {e}]"

    # 4. Post-processing
    if "not stated in context" in final_answer.lower():
        # Fallback check: if the answer is just "Not stated in context", keep it as is.
        pass
    
    # 5. Hallucination / Consistency Check (Keeping logic from 1.9.1)
    # Check if sentences in the answer are actually supported by the context
    def is_supported_sentence(sentence: str, support_source_text: str, model_threshold: float = support_threshold) -> (bool, float):
        try:
            emb_model = get_embed_model()
            sent_emb = emb_model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
            # We compare against the original context chunks if available, but here we check against combined
            # To be efficient, we check against the combined text chunks
            candidates = [support_source_text]
            cand_embs = emb_model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
            sims = (cand_embs @ sent_emb).tolist()
            max_sim = max(sims) if sims else 0.0
            max_sim_scaled = round((float(max_sim) + 1) / 2, 3)
            return (max_sim_scaled >= model_threshold, max_sim_scaled)
        except Exception as e:
            return (False, 0.0)

    # Only run support check if it's a real answer
    if "not stated" not in final_answer.lower() and len(final_answer) > 10:
        sentence_candidates = [s.strip() for s in re.split(r'(?<=[.!?])\s+', final_answer) if s.strip()]
        for s in sentence_candidates:
            supported, score = is_supported_sentence(s, combined_context, model_threshold=support_threshold)
            if not supported:
                # If consistency is low, we might flag it, but for now we return the model output
                # with a visual warning in the UI handled below by cosine score
                pass

    # 6. Calculate Metrics (Cosine, ROUGE)
    cosine_score = None
    try:
        emb_model = get_embed_model()
        text_emb = emb_model.encode([safe_context], convert_to_numpy=True, normalize_embeddings=True)[0]
        summary_emb = emb_model.encode([final_answer], convert_to_numpy=True, normalize_embeddings=True)[0]
        cosine_sim = float(np.dot(text_emb, summary_emb))
        cosine_score = round((cosine_sim + 1) / 2, 3)
    except Exception:
        cosine_score = 0.0

    rouge2_score = None
    rougeL_score = None
    try:
        scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(safe_context, final_answer)
        rouge2_score = round(scores['rouge2'].fmeasure, 3)
        rougeL_score = round(scores['rougeL'].fmeasure, 3)
    except Exception:
        pass
    
    # Render the Small Table HTML (Same as 1.9.1)
    def rating_label(score, high_thr, med_thr):
        if score is None: return ("-", "-", "-")
        if score >= high_thr: return (f"{score:.3f}", "-", "-")
        elif score >= med_thr: return ("-", f"{score:.3f}", "-")
        else: return ("-", "-", f"{score:.3f}")

    cosine_high, cosine_med = 0.85, 0.70
    rouge_high, rouge_med = 0.6, 0.4
    cos_high, cos_med, cos_low = rating_label(cosine_score, cosine_high, cosine_med)
    r2_high, r2_med, r2_low = rating_label(rouge2_score, rouge_high, rouge_med)
    rL_high, rL_med, rL_low = rating_label(rougeL_score, rouge_high, rouge_med)

    with st.expander("📊 Evaluation Metrics Summary", expanded=False):
        small_table_html = f"""
        <style>
            .small-table {{ border-collapse: collapse; width: 60%; font-size: 13px; margin-top: 6px; }}
            .small-table th, .small-table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
            .small-table th {{ background-color: #f9f9f9; font-weight: bold; }}
            .small-table td {{ font-family: monospace; }}
            .green {{ color: green; font-weight: bold; }}
            .orange {{ color: orange; font-weight: bold; }}
            .red {{ color: red; font-weight: bold; }}
        </style>
        <table class='small-table'>
            <tr><th>Metric</th><th>Description</th><th>High (🟢)</th><th>Medium (🟠)</th><th>Low (🔴)</th></tr>
            <tr><td><b>Cosine Similarity</b></td><td>Context Similarity</td><td class='green'>{cos_high}</td><td class='orange'>{cos_med}</td><td class='red'>{cos_low}</td></tr>
            <tr><td><b>ROUGE-2</b></td><td>Bigram Overlap with Context</td><td class='green'>{r2_high}</td><td class='orange'>{r2_med}</td><td class='red'>{r2_low}</td></tr>
            <tr><td><b>ROUGE-L</b></td><td>Longest Common Sequence Match</td><td class='green'>{rL_high}</td><td class='orange'>{rL_med}</td><td class='red'>{rL_low}</td></tr>
        </table>
        """
        st.markdown(small_table_html, unsafe_allow_html=True)
    
    total_time = time.perf_counter() - start_time
    st.success(f"✓ Answer generated in **{total_time:.2f} seconds**.")

    return final_answer, cosine_score, rouge2_score, rougeL_score

# =============================================
# add_document_to_topic (store chunking strategy in metadata)
# =============================================
def add_document_to_topic(topic_name: str, pdf_path: str, pdf_filename: str, full_text: str, table_text: str,
                          chunk_strategy: str = "recursive", chunk_size: int = 1000, overlap: int = 200):
    st.write("🔹 Starting document ingestion...")
    sub_topic = derive_subtopic_from_filename(pdf_filename)
    st.info(f"Detected sub-topic: **{sub_topic}**")
    col = get_or_create_collection(topic_name)
    combined_text = full_text
    if table_text.strip():
        combined_text += "\n\n[TABLES]\n\n" + table_text
    st.write(f"Length of combined text: {len(combined_text)} characters")
    chunks = chunk_text(combined_text, chunk_size=chunk_size, overlap=overlap, strategy=chunk_strategy)
    st.write(f"✓ Chunking complete — {len(chunks)} chunks created using strategy **{chunk_strategy}**.")
    if not chunks:
        return {"error": "No text chunks extracted."}
    st.write("🧬 Generating embeddings...")
    embs = embed_texts(chunks)
    st.write(f"✓ Embeddings generated — shape: {embs.shape}")
    timestamp = int(time.time())
    ids = [f"{pdf_filename}__{i}_{timestamp}" for i in range(len(chunks))]
    metadatas = [
        {
            "pdf_filename": pdf_filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "chunk_index": i,
            "topic": topic_name,
            "sub_topic": sub_topic,
            "chunking_strategy": chunk_strategy,
        }
        for i in range(len(chunks))
    ]
    st.write("⎌ Adding to Chroma collection...")
    col.add(documents=chunks, embeddings=embs.tolist(), ids=ids, metadatas=metadatas)
    st.write("✓ Added to Chroma (PersistentClient).")
    backup_chroma() # backup chromadb
    return {"count": len(chunks), "ids": ids, "collection": topic_name, "sub_topic": sub_topic, "chunking_strategy": chunk_strategy}

# =============================================
# query_collection (unchanged)
# =============================================
def query_collection(topic: str, query: str, top_k=3, relevance_threshold=0.35, sub_topic: str = None):
    """
    Query a Chroma collection and optionally filter by sub_topic.
    If sub_topic is None or equals 'All' -> no where filter is applied.
    """                                                       
    col = get_or_create_collection(topic)
    q_embs = embed_texts([query])
    if q_embs.size == 0:
        return []
    q_emb = q_embs[0].tolist()
    where = None
    if sub_topic and sub_topic.lower() != "all":
        where = {"sub_topic": sub_topic}
    results = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"], # in chromaDB "ids" are always included
        where=where
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]
    relevant = []
    for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
        if dist <= relevance_threshold:
            meta_sub = meta.get("sub_topic") if isinstance(meta, dict) else None
            if not meta_sub:
                meta_sub = "general"
                if isinstance(meta, dict):
                    meta["sub_topic"] = meta_sub
            relevant.append({"id": id_, "text": doc, "metadata": meta, "distance": dist})
    return sorted(relevant, key=lambda x: x["distance"])

# =============================================
# Simple login gate helper (per-tab)
# =============================================
def tab_login_gate(tab_key: str, display_name: str) -> bool:
    """
    Simple login gate. Returns True if authenticated for this tab_key, otherwise shows UI and returns False.
    tab_key: short string used for session key, e.g. "upload" or "manage"
    Also stores user role in st.session_state[f"role_{tab_key}"] ('admin' or 'viewer').
    """
    sess_key = f"auth_{tab_key}"
    role_key = f"role_{tab_key}"

    if st.session_state.get(sess_key, False):
        col1, col2 = st.columns([1, 5])
        role = st.session_state.get(role_key, "admin")
        with col1:
            if st.button(f"⏻ Logout {display_name}", key=f"logout_{tab_key}"):
                st.session_state[sess_key] = False
                st.session_state.pop(role_key, None)
                st.toast("Logged out.")
                st.rerun()
        with col2:
            st.markdown(f"**Authenticated for {display_name}** ({role})")
        return True

    st.info(f"🔐 {display_name} requires login.")
    user = st.text_input(f"{display_name} - UserID:", key=f"user_{tab_key}")
    pwd = st.text_input(f"{display_name} - Password:", key=f"pass_{tab_key}", type="password")
    if st.button(f"⇥ Login to {display_name}", key=f"login_{tab_key}"):
        # Check Admin
        if (user == ADMIN_USER) and (pwd == ADMIN_PASS):
            st.session_state[sess_key] = True
            st.session_state[role_key] = "admin"
            st.success("Login successful (Admin).")
            st.rerun()
        # Check Viewer
        elif (user == VIEWER_USER) and (pwd == VIEWER_PASS):
            st.session_state[sess_key] = True
            st.session_state[role_key] = "viewer"
            st.success("Login successful (Viewer).")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return False

# =============================================
# UI: Radio-based "Tabs" (Conditional Rendering for Independent Access)
# =============================================
# Use st.radio for tab selection (renders only selected content — fixes execution bleed)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Ask Questions"  # Default to public tab

# The main navigation is updated here: "Manage Topics" -> "DocIQ-Admin"
# Added "DocIQ Overview" and "User-Guide"
SIDEBAR_OPTIONS = ["Ask Questions", "Upload PDFs", "DocIQ-Admin", "User-Guide", "DocIQ Overview", "Knowledge Catalogue"]
if st.session_state.active_tab == "Manage Topics": # Handle old state for continuity
    st.session_state.active_tab = "DocIQ-Admin"

# Handle potential state mismatch if new options added to list
if st.session_state.active_tab not in SIDEBAR_OPTIONS:
    st.session_state.active_tab = "Ask Questions"

selected_tab = st.sidebar.radio(
    "Select Section:",
    SIDEBAR_OPTIONS,
    index=SIDEBAR_OPTIONS.index(st.session_state.active_tab),
    key="tab_selector")

st.session_state.active_tab = selected_tab

# -----------------------
# DocIQ Overview Tab
# -----------------------
if selected_tab == "DocIQ Overview":
    st.session_state["active_tab"] = "DocIQ Overview"
    st.header("DocIQ Overview")
    
    # Calculate absolute path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "docIQ_overview.md")
    
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.markdown(md_content, unsafe_allow_html=True)
        else:
            st.warning(f"File 'docIQ_overview.md' not found at expected path: {file_path}")
    except Exception as e:
        st.error(f"Error reading 'docIQ_overview.md': {e}")

# -----------------------
# User-Guide Tab
# -----------------------
if selected_tab == "User-Guide":
    st.session_state["active_tab"] = "User-Guide"
    st.header("User Guide")
    
    # Calculate absolute path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "docIQ_userguide.md")
    
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.markdown(md_content, unsafe_allow_html=True)
        else:
            st.warning(f"File 'docIQ_userguide.md' not found at expected path: {file_path}")
    except Exception as e:
        st.error(f"Error reading 'docIQ_userguide.md': {e}")

# -------------------------------------
# Knowledge Catalogue Overview Tab
# -------------------------------------
if selected_tab == "Knowledge Catalogue":
    st.session_state["active_tab"] = "Knowledge Catalogue"
    st.header("Knowledge Catalogue")
    
    # Calculate absolute path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "knowledge_catalog.md")
    
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.markdown(md_content, unsafe_allow_html=True)
        else:
            st.warning(f"File 'knowledge_catalog.md' not found at expected path: {file_path}")
    except Exception as e:
        st.error(f"Error reading 'knowledge_catalog.md': {e}")
        
# -----------------------
# Upload tab (gated) - Only renders if selected
# -----------------------
if selected_tab == "Upload PDFs":
    st.session_state["active_tab"] = "Upload PDFs"
    # Login gate for this tab
    if not tab_login_gate("upload", "Upload PDFs"):
        st.stop()
    
    # Check role
    current_role = st.session_state.get("role_upload", "admin")
    is_viewer = (current_role == "viewer")

    st.header("Upload a PDF with text, tables, or charts/images")
    if is_viewer:
        st.info("ℹ You are logged in as Viewer. Uploads are disabled.")

    try:
        existing_collections = [c.name for c in chroma_client.list_collections()]
    except Exception:
        existing_collections = []

    topic_mode = st.radio("Select how you want to assign a topic:", ["Select existing topic", "Create new topic"])
    topic = None
    if topic_mode == "Select existing topic":
        if existing_collections:
            topic = st.selectbox("Choose an existing topic:", existing_collections)
        else:
            st.info("No existing topics found. Please create a new one.")
    else:
        topic = st.text_input("Enter a new topic name:", value="")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    use_ocr = st.checkbox("Use OCR for images/charts if text not found", value=True)

    # chunking strategy selection
    st.markdown("**Choose chunking strategy to use for this document**")

    chunk_option = st.radio(
        "Chunking strategy:",
        ["Adaptive Chunking (NLTK)", "Recursive Sentence Chunking (Regex)", "Sliding Window Chunking"],
        index=0,
        help="Adaptive = NLTK-based intelligent sentence grouping; Recursive = Regex-based; Sliding-Window = fixed size."
    )

    # Container to group dynamic UI controls
    chunk_controls = st.container()

    with chunk_controls:
        if "Adaptive" in chunk_option:
            # Adaptive NLTK
            st.markdown("#### Adaptive Chunking (NLTK) Settings")
            user_chunk_size = st.number_input(
                "Target Chunk Size (characters)",
                min_value=300, max_value=8000, value=1200, step=100,
                help="Target size. The chunker groups sentences until this limit is reached."
            )
            user_overlap = st.number_input(
                "Overlap (characters)",
                min_value=0, max_value=2000, value=200, step=50,
                help="Character count of sentences carried over from the end of the previous chunk."
            )
        elif "Recursive" in chunk_option:
            st.markdown("#### Recursive Sentence Chunking Settings")
            user_chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=300, max_value=8000, value=1500, step=100,
                help="Upper size limit of each chunk. The recursive chunker splits on sentence boundaries and merges until this limit."
            )
            user_overlap = st.number_input(
                "Overlap (characters)",
                min_value=0, max_value=2000, value=200, step=50,
                help="How many characters from the end of the previous chunk are carried forward into the next chunk for context continuity."
            )
        else:
            # Sliding Window
            st.markdown("#### Sliding Window Chunking Settings")
            user_chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=300, max_value=8000, value=1000, step=100,
                help="Fixed window size used for chunking. This was the original chunking method in your app."
            )
            user_overlap = st.number_input(
                "Overlap (characters)",
                min_value=0, max_value=2000, value=200, step=50,
                help="Characters included from the previous window to maintain continuity."
            )

    # Map Selection to internal string key
    if "Adaptive" in chunk_option:
        chunk_strategy_internal = "adaptive"
    elif "Recursive" in chunk_option:
        chunk_strategy_internal = "recursive"
    else:
        chunk_strategy_internal = "fixed"

    # Disable Upload button if viewer
    if st.button("Upload and Process", disabled=is_viewer):
        if not uploaded_file or not topic.strip():
            st.warning("Please upload a PDF and select or create a topic name.")
        else:
            pdf_bytes = uploaded_file.read()
            pdf_filename = uploaded_file.name
            pdf_path = os.path.join(UPLOADS_DIR, f"{int(time.time())}_{pdf_filename}")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf_bytes(pdf_bytes)
                if not text.strip() and use_ocr:
                    st.info("No text found in PDF, trying OCR...")
                    text = extract_text_with_ocr(pdf_bytes)

            with st.spinner("Extracting tables..."):
                try:
                    table_text = extract_tables_as_text(pdf_path)
                except Exception as e:
                    table_text = ""
                    st.warning(f"Table extraction failed: {e}")

            if not text.strip() and not table_text.strip():
                st.error("❌ No text or tables found in PDF.")
            else:
                with st.spinner("Chunking, embedding, and storing..."):
                    result = add_document_to_topic(topic.strip(), pdf_path, pdf_filename, text, table_text,
                                                   chunk_strategy=chunk_strategy_internal,
                                                   chunk_size=int(user_chunk_size), overlap=int(user_overlap))

                st.subheader("Upload Result")
                st.json(result)

                if "error" not in result:
                    st.success(f"✓ Added {result['count']} chunks to collection '{topic.strip()}'. (strategy: {result.get('chunking_strategy')})")
                    col = get_or_create_collection(topic.strip())
                    try:
                        col_size = col.count()
                    except Exception:
                        col_size = "unknown"
                    st.write(f"Collection total chunks: {col_size}")

                    try:
                        data = col.get(include=["documents", "metadatas"])  # Chroma does NOT allow "ids" inside include[], "id" are included
                        docs = data.get("documents", [])
                        metas = data.get("metadatas", [])
                        ids = data.get("ids", [])
                        st.subheader("Sample Chunks (First 3)")
                        for i in range(min(3, len(docs))):
                            st.markdown(f"**ID:** `{ids[i]}`")
                            st.write("Metadata:", metas[i])
                            st.write("Excerpt:", docs[i][:500])
                            st.markdown("---")
                    except Exception as e:
                        st.warning(f"Could not fetch sample chunks: {e}")

# -----------------------
# Ask Questions tab (PUBLIC - NO LOGIN)
# -----------------------
if selected_tab == "Ask Questions":
    st.session_state["active_tab"] = "Ask Questions"
    st.markdown('<h1 class="header1">Query your knowledge base</h1>', unsafe_allow_html=True)

    MAX_INACTIVE_MINUTES = 15
    now_ts = time.time()
    last_active = st.session_state.get("last_active_time", now_ts)
    if now_ts - last_active > MAX_INACTIVE_MINUTES * 60:
        st.session_state.clear()
        st.info(f"🕒 Session cleared due to **{MAX_INACTIVE_MINUTES}** minutes of inactivity.")
    st.session_state["last_active_time"] = now_ts

    try:
        available_topics = [c.name for c in chroma_client.list_collections()]
    except Exception:
        available_topics = []

    if not available_topics:
        st.warning("No topics found. Please upload a document first.")
    else:
        topic_q = st.selectbox("Select topic to query:", available_topics, key="query_topic")
        subtopic_options = ["All"]
        try:
            col_for_topic = get_or_create_collection(topic_q)
            try:
                data_meta = col_for_topic.get(include=["metadatas"])
                metas_all = data_meta.get("metadatas", [])
                sth = set()
                for m in metas_all:
                    if isinstance(m, dict):
                        stv = m.get("sub_topic") or "general"
                        sth.add(stv)
                subtopic_options = ["All"] + sorted([s for s in sth if s and s.lower() != "all"])
            except Exception:
                subtopic_options = ["All"]
        except Exception:
            subtopic_options = ["All"]

        selected_subtopic = st.selectbox("Select sub-topic (optional):", subtopic_options, index=0, key="query_subtopic")

        query_text = st.text_input("Enter your question:", key="query_text", value=st.session_state.get("query_text", ""), placeholder="Type your new question here...")

        relevance_threshold = st.slider(
            "Set relevance threshold (lower = more strict, higher = more inclusive)",
            min_value=0.1, max_value=1.0, value=0.6, step=0.05
        )
        top_k = st.slider("Number of top results to retrieve (top_k)", min_value=1, max_value=20, value=5, step=1)

        fuzzy_threshold = st.slider(
            "Fuzzy match threshold for cached questions (0.0–1.0). Higher = stricter exactness",
            min_value=0.5, max_value=0.95, value=0.78, step=0.01
        )

        if st.button("Search"):
            if not query_text.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    topic_norm = topic_q.strip()
                    cache_start = time.time()
                    best = get_best_fuzzy_match(topic_norm, query_text.strip(), threshold=fuzzy_threshold)
                    cache_elapsed = time.time() - cache_start

                    if best:
                        # cached path
                        # best = (rid, stored_q, stored_a, created_at, fb, sub_topic, cos_m, r2_m, rL_m, score, source_docs, rel_th, k)
                        rid, stored_q, stored_a, created_at, fb, sub_topic, cos_m, r2_m, rL_m, score, source_docs, rel_th, k = best
                        st.session_state["last_query"] = {
                            "topic": topic_norm,
                            "sub_topic": sub_topic or "all",
                            "question": query_text.strip(),
                            "cached": True,
                            "entry_id": rid,
                            "answer": stored_a,
                            "feedback_status": fb,
                            "meta": {
                                "score": score,
                                "created_at": created_at,
                                "cache_time": cache_elapsed,
                                "cosine_score": cos_m,
                                "rouge2_score": r2_m,
                                "rougeL_score": rL_m
                            },
                            "source_docs": source_docs or []
                        }
                    else:
                        vec_start = time.time()
                        # Use subtopic filter when selected_subtopic != "All"                                                     
                        subtopic_filter = selected_subtopic if selected_subtopic and selected_subtopic != "All" else None
                        docs = query_collection(topic_norm, query_text.strip(), top_k=top_k, relevance_threshold=relevance_threshold, sub_topic=subtopic_filter)
                        vec_elapsed = time.time() - vec_start

                        if not docs:
                            st.warning("No relevant results found. Try using a higher threshold (more inclusive) or choose 'All' sub-topics.")
                        else:
                            st.subheader("Top Results")
                            # Build DataFrame including sub_topic column from metadata                                                          
                            rows_for_df = []
                            for d in docs:
                                meta = d.get("metadata") or {}
                                sub_t = meta.get("sub_topic") if isinstance(meta, dict) else "general"
                                rows_for_df.append([d["id"], sub_t, str(meta), d["text"][:500].replace("\n", " "), f"{d['distance']:.4f}"])
                            df = pd.DataFrame(rows_for_df, columns=["ID", "sub_topic", "Metadata", "Excerpt", "Distance"])
                            st.dataframe(df, use_container_width=True)

                            st.subheader("Generative Answer")
                            # combined_text = " ".join([d["text"] for d in docs])
                            # Use joined context of retrieved chunks
                            combined_text = "\n\n".join([d["text"] for d in docs])

                            sum_start = time.perf_counter()
                            with st.spinner("Generating answer from context..."):
                                summary, cosine_score, rouge2_score, rougeL_score = generate_answer_from_context(combined_text, question=query_text.strip(), top_k=top_k)
                            sum_elapsed = time.perf_counter() - sum_start

                            # derive source document filenames from ids
                            # IDs follow pattern: "{pdf_filename}__{i}_{timestamp}"
                            source_docs = []
                            for d in docs:
                                try:
                                    raw_id = d.get("id", "")
                                    if "__" in raw_id:
                                        pdfname = raw_id.split("__", 1)[0]
                                        source_docs.append(pdfname)
                                except Exception:
                                    pass
                            source_docs = sorted(list(dict.fromkeys([s for s in source_docs if s])))  # unique preserve order

                            cache_subtopic_value = subtopic_filter if subtopic_filter else "all"
                            inserted_id = insert_cache(topic_norm, query_text.strip(), summary, feedback_status=None,
                                                       sub_topic=cache_subtopic_value,
                                                       cosine_score=cosine_score,
                                                       rouge2_score=rouge2_score,
                                                       rougeL_score=rougeL_score,
                                                       source_docs=source_docs,
                                                       relevance_threshold=relevance_threshold,
                                                       top_k=top_k)

                            st.session_state["last_query"] = {
                                "topic": topic_norm,
                                "sub_topic": cache_subtopic_value,
                                "question": query_text.strip(),
                                "cached": False,
                                "entry_id": inserted_id,
                                "answer": summary,
                                "feedback_status": None,
                                "meta": {
                                    "vec_time": vec_elapsed,
                                    "sum_time": sum_elapsed,
                                    "cosine_score": cosine_score,
                                    "rouge2_score": rouge2_score,
                                    "rougeL_score": rougeL_score
                                },
                                # attach source docs so we can display the message later
                                "source_docs": source_docs
                            }

                    st.session_state["question_count"] = st.session_state.get("question_count", 0) + 1

                except Exception as e:
                    st.error(f"Query failed: {e}")

        # ============================================================== display last_query if present
        if "last_query" in st.session_state:
            last = st.session_state["last_query"]
            rid = last["entry_id"]

            st.subheader(f"Answer for: {last['question']}  — *(Topic: {last['topic']}, Sub-topic: {last.get('sub_topic','all')})*")
            st.write(last["answer"])

            # Show metrics if present
            meta_metrics = last.get("meta", {})
            if any(k in meta_metrics for k in ["cosine_score", "rouge2_score", "rougeL_score"]):
                metrics_html = f"""
                <div style="display:flex; gap:12px; align-items:center;">
                    <div><b>Cosine:</b> {meta_metrics.get('cosine_score', '-')}</div>
                    <div><b>ROUGE-2:</b> {meta_metrics.get('rouge2_score', '-')}</div>
                    <div><b>ROUGE-L:</b> {meta_metrics.get('rougeL_score', '-')}</div>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)

            if last["cached"]:
                st.caption(f"⚡ Retrieved from cache (score={last['meta']['score']:.3f}, cached at {last['meta']['created_at']})")
            else:
                st.caption(f"Vector fetch took: **{last['meta']['vec_time']:.2f}s**, Generation took: **{last['meta']['sum_time']:.2f}s**")

            # --- Show persistent source documents if present
            srcs = last.get("source_docs") or []
            if srcs:
                # display JSON list (as instructed)
                st.info(
    f"This answer was generated from document(s): {json.dumps(srcs)}\n\n"
    "⚠ *Note: This answer is generated by an AI model based on the retrieved content. "
    "Always check original document(s).*"
)

            def show_feedback_controls_and_apply(entry_id: int):
                col_yes, col_no = st.columns(2)
                def apply_feedback(status):
                    try:
                        with sqlite3.connect(CACHE_DB_PATH) as conn:
                            conn.execute("UPDATE qa_cache SET feedback_status = ? WHERE id = ?", (status, entry_id))
                            conn.commit()
                            backup_sqlite() #back feedback_status changes
                        st.session_state["last_query"]["feedback_status"] = status
                        st.toast(f"✓ Feedback updated → {status}")
                    except Exception as e:
                        st.error(f"Update failed: {e}")
                with col_yes:
                    if st.button("👍 Helpful (mark Y)", key=f"fb_yes_{entry_id}"):
                        apply_feedback("Y")
                with col_no:
                    if st.button("👎 Not helpful (mark N)", key=f"fb_no_{entry_id}"):
                        apply_feedback("N")
                with sqlite3.connect(CACHE_DB_PATH) as conn:
                    cur = conn.execute("SELECT feedback_status FROM qa_cache WHERE id = ?", (entry_id,))
                    val = cur.fetchone()
                fb_now = val[0] if val and val[0] else "Pending"
                st.caption(f"Current feedback status: **{fb_now}**")

            show_feedback_controls_and_apply(rid)

            def reset_query():
                for key in ["last_query", "query_text", "final_summary",
                            "rouge2_score", "rougeL_score", "cosine_score"]:
                    st.session_state.pop(key, None)
                st.session_state["query_text"] = ""
                st.session_state["active_tab"] = "Ask Questions"
                st.toast("🧹 Cleared previous question. Ready for a new query!")
                st.session_state["should_rerun"] = True

            if st.session_state.get("question_count", 0) >= 1:
                st.markdown("---")
                st.button("✧ Ask another question", on_click=reset_query)

# -----------------------
# DocIQ-Admin tab (gated) - UPDATED NAVIGATION
# -----------------------
if selected_tab == "DocIQ-Admin":
    st.session_state["active_tab"] = "DocIQ-Admin"
    # Login gate for this tab
    if not tab_login_gate("admin", "DocIQ-Admin"):
        st.stop()
    
    # Check role
    current_role = st.session_state.get("role_admin", "admin")
    is_viewer = (current_role == "viewer")

    st.header("⚿ DocIQ-Admin Console")
    
    # --- New Sub-Menu Radio Selection (3 options) ---
    ADMIN_SUB_MENU_OPTIONS = ["Chunking Strategies Overview", "Manage Topics", "Manage Cached DB"]
    admin_sub_menu = st.radio(
        "Admin Task:",
        ADMIN_SUB_MENU_OPTIONS,
        key="admin_sub_menu_radio",
        horizontal=True
    )

    # --- 1. Chunking Strategies Overview (formerly nested) ---
    if admin_sub_menu == "Chunking Strategies Overview":
        
        st.subheader("⌗ Chunking Strategies")
        
        st.markdown("View which chunking strategy was used for each uploaded document and how many vector records exist per document.")
        # Button enabled for viewer
        if st.button("View Chunking Strategies"):
            try:
                all_collections = [c.name for c in chroma_client.list_collections()]
                rows = []
                for col_name in all_collections:
                    col = get_or_create_collection(col_name)
                    data = col.get(include=["metadatas"]) # "ids" always included in ChromaDB
                    metas = data.get("metadatas", [])
                    # ids = data.get("ids", []) # not needed for this overview
                    for meta in metas:
                        if not isinstance(meta, dict):
                            continue
                        pdf = meta.get("pdf_filename", "unknown")
                        subtopic = meta.get("sub_topic", "unknown")
                        strategy = meta.get("chunking_strategy", "unknown")
                        rows.append((pdf, col_name, subtopic, strategy))
                if not rows:
                    st.warning("No chunking metadata found (No collections available in Chroma DB).")
                else:
                    df = pd.DataFrame(rows, columns=["Document", "Topic", "Subtopic", "Strategy"])
                    grouped = df.groupby(["Document", "Topic", "Subtopic", "Strategy"]).size().reset_index(name="Total Records")
                    grouped = grouped.sort_values(["Document", "Topic", "Subtopic"])
                    st.dataframe(grouped, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to build chunking strategies overview: {e}")
        
        st.markdown("---")
        st.info("Select 'Manage Topics' to delete vector collections or 'Manage Cached DB' to clear the Q&A log.")


    # --- 2. Manage Topics (Vector/Chroma Management) ---
    elif admin_sub_menu == "Manage Topics":
        
        try:
            all_topics = [c.name for c in chroma_client.list_collections()]
        except Exception:
            all_topics = []

        if "delete_entire_topic" not in st.session_state:
            st.session_state.delete_entire_topic = None
        if "total_before" not in st.session_state:
            st.session_state.total_before = None
        if "delete_sub_topics" not in st.session_state:
            st.session_state.delete_sub_topics = []
        if "total_before_sub" not in st.session_state:
            st.session_state.total_before_sub = 0
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = False
        if "confirm_delete_subs" not in st.session_state:
            st.session_state.confirm_delete_subs = False

        if not all_topics:
            st.warning("No topics available in Chroma DB to manage.")
        else:
            st.subheader("🗑 Manage topics and sub-topics")
            topic_to_manage = st.selectbox("🗑 Select a topic to delete:", all_topics, key="manage_topic_select")
            try:
                col = get_or_create_collection(topic_to_manage)
                metas_all = col.get(include=["metadatas"]).get("metadatas", [])
                sublist = []
                for m in metas_all:
                    if isinstance(m, dict):
                        sublist.append(m.get("sub_topic", "general"))
                sublist_unique = sorted(set([s or "general" for s in sublist]))
            except Exception:
                sublist_unique = []
            st.write("Sub-topics under this topic:", ", ".join(sublist_unique) if sublist_unique else "— none found —")
            delete_option = st.radio("Delete options:", ["Delete entire topic", "Delete select sub-topic(s)"], key="delete_option_radio")
            
            # Reset delete state variables on sub-menu change
            # Note: This reset logic is now partially redundant due to the new structure, but safe to keep 
            # or simplify down to managing the confirmation flags.

            if delete_option == "Delete entire topic":
                st.write("This will delete the entire Chroma collection and all its records.")
                # Disabled for viewer
                if st.button("Prepare to delete entire topic", disabled=is_viewer):
                    try:
                        before_count = col.count()
                    except Exception:
                        before_count = "unknown"
                    st.session_state.delete_entire_topic = topic_to_manage
                    st.session_state.total_before = before_count
                    st.session_state.confirm_delete = True
                    st.session_state.confirm_delete_subs = False # Ensure other confirmation is off

                if st.session_state.get("confirm_delete", False) and st.session_state.delete_entire_topic == topic_to_manage:
                    st.warning(f"About to delete topic **{st.session_state.delete_entire_topic}** with **{st.session_state.total_before}** records.")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✓ Confirm Delete Entire Topic"):
                            with st.spinner(f"Deleting topic '{topic_to_manage}'..."):
                                try:
                                    chroma_client.delete_collection(name=topic_to_manage)
                                    backup_chroma() # backup chromadb latest changes 
                                    time.sleep(0.5)
                                except Exception as e:
                                    st.error(f"Deletion failed: {e}")
                                    st.stop()
                            try:
                                remaining_topics = [c.name for c in chroma_client.list_collections()]
                                after_exists = topic_to_manage in remaining_topics
                            except Exception:
                                after_exists = None
                            audit_data = {
                                "Topic Name": [topic_to_manage],
                                "Records Before Delete": [st.session_state.total_before],
                                "Records After Delete": [0 if not after_exists else "Still Exists"],
                                "Deletion Timestamp": [datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")]
                            }
                            audit_df = pd.DataFrame(audit_data)
                            st.subheader("🧾 Deletion Audit Report")
                            st.dataframe(audit_df, use_container_width=True)
                            if not after_exists:
                                st.success(f"✓ Topic '{topic_to_manage}' successfully deleted.")
                            else:
                                st.error(f"⨯ Deletion failed — topic still exists.")
                            st.session_state.confirm_delete = False
                            st.session_state.delete_entire_topic = None
                            st.session_state.total_before = None
                            #st.rerun() # Refresh topic list # commented to view  "Deletion Audit Report"
                    with c2:
                        if st.button("✗ Cancel Delete Entire Topic"):
                            st.session_state.confirm_delete = False
                            st.session_state.delete_entire_topic = None
                            st.session_state.total_before = None
                            st.info("Deletion cancelled. Selections cleared.")
            else:
                st.write("🗑 Choose one or more sub-topic(s) to delete only their documents (keeps other sub-topics intact).")
                if sublist_unique:
                    chosen_subs = st.multiselect("Select sub-topic(s) to delete:", sublist_unique, key="manage_select_subs")
                else:
                    chosen_subs = []
                # Disabled for viewer
                if st.button("Delete Selected sub-topic(s)", disabled=is_viewer):
                    if not chosen_subs:
                        st.warning("Please select at least one sub-topic to delete.")
                    else:
                        total_before_sub = 0
                        for ss in chosen_subs:
                            try:
                                data = col.get(include=["metadatas"], where={"sub_topic": ss})
                                ids_for = data.get("ids", [])
                                total_before_sub += len(ids_for)
                            except Exception:
                                pass
                        st.session_state.delete_sub_topics = chosen_subs
                        st.session_state.total_before_sub = total_before_sub
                        st.session_state.confirm_delete_subs = True
                        st.session_state.confirm_delete = False # Ensure other confirmation is off

                if st.session_state.get("confirm_delete_subs", False) and st.session_state.delete_sub_topics:
                    st.warning(f"About to delete sub-topic(s) {', '.join(st.session_state.delete_sub_topics)} containing {st.session_state.total_before_sub} records.")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✓ Confirm Delete Selected sub-topic(s)"):
                            deleted_counts = {}
                            for ss in list(st.session_state.delete_sub_topics):
                                try:
                                    data_before = col.get(include=["metadatas"], where={"sub_topic": ss})
                                    ids_before = data_before.get("ids", [])
                                    before_ct = len(ids_before)
                                    try:
                                        col.delete(where={"sub_topic": ss}) #scoped delete using collection object (col)
                                        backup_chroma() #update/sync  backup after deletion of sub-topic(s)
                                    except Exception:
                                        if ids_before:
                                            col.delete(ids=ids_before)
                                    deleted_counts[ss] = before_ct                                    
                                except Exception as e:
                                    deleted_counts[ss] = f"error: {e}"
                            audit_data = {
                                "Sub-topic": list(deleted_counts.keys()),
                                "Records Deleted": list(deleted_counts.values()),
                                "Deletion Timestamp": [datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")] * len(deleted_counts)
                            }
                            audit_df = pd.DataFrame(audit_data)
                            st.subheader("🧾 Sub-topic Deletion Audit Report")
                            st.dataframe(audit_df, use_container_width=True)
                            st.success("✓ Selected sub-topic(s) deleted. See audit for details.")
                            st.session_state.delete_sub_topics = []
                            st.session_state.total_before_sub = 0
                            st.session_state.confirm_delete_subs = False
                            #st.rerun() # Refresh sub-topic list # commented to view  "Deletion Audit Report"
                    with c2:
                        if st.button("✗ Cancel Delete Selected sub-topic(s)"):
                            st.session_state.delete_sub_topics = []
                            st.session_state.total_before_sub = 0
                            st.session_state.confirm_delete_subs = False
                            st.info("Deletion cancelled. Selections cleared.")


    # --- 3. Manage Cached DB (SQLite Management) ---
    elif admin_sub_menu == "Manage Cached DB":
        st.subheader("View / Delete cached Q&A entries (cache DB)")
        try:
            topics_in_cache = _cache_conn.execute("SELECT DISTINCT topic FROM qa_cache ORDER BY topic").fetchall()
            topics_in_cache = [t[0] for t in topics_in_cache]
        except Exception:
            topics_in_cache = []

        if not topics_in_cache:
            st.warning("No cached topics in cache DB yet.")
        else:
            if "manage_topic_open" not in st.session_state:
                st.session_state.manage_topic_open = {}
            if "manage_filter_open" not in st.session_state:
                st.session_state.manage_filter_open = {}

            for t in topics_in_cache:
                cols = st.columns([6, 2, 4])
                cols[0].write(f"**{t}**")
                manage_key = f"manage_qna_{t}"
                filter_key = f"filter_qna_{t}"
                if cols[1].button("Manage Q&A", key=manage_key):
                    st.session_state.manage_topic_open[t] = not st.session_state.manage_topic_open.get(t, False)
                if cols[2].button("Filter & Sort cached Q&A by metric values", key=filter_key):
                    st.session_state.manage_filter_open[t] = not st.session_state.manage_filter_open.get(t, False)

                if st.session_state.manage_topic_open.get(t, False):
                    with st.expander(f"Manage Q&A — Full list for topic: {t}", expanded=True):
                        rows = fetch_cache_by_topic(t, only_helpful=False)
                        if not rows:
                            st.info("No cached Q&A for this topic.")
                        else:
                            df_rows = []
                            for row in rows:
                                # row: (id, q, a, created, fb, sub, cos, r2, rL, docs, rel_th, k)
                                rid, q_text, a_text, created_at, fb, sub_topic, cos_m, r2_m, rL_m, source_docs_list, rel_th, k = row
                                df_rows.append({
                                    "id": rid,
                                    "question": q_text,
                                    "answer": a_text,
                                    "created_at": created_at,
                                    "feedback_status": fb or "",
                                    "sub_topic": sub_topic or "all",
                                    "cosine_score": cos_m,
                                    "rouge2_score": r2_m,
                                    "rougeL_score": rL_m,
                                    "source_docs": json.dumps(source_docs_list),
                                    "relevance_threshold": rel_th,
                                    "top_k": k
                                })
                            df_cache = pd.DataFrame(df_rows)
                            st.markdown("**Individual cached entries (full list)**")
                            for _, r in df_cache.sort_values(by="created_at", ascending=False).iterrows():
                                rid = int(r["id"])
                                q_text = r["question"]
                                a_text = r["answer"]
                                created_at = r["created_at"]
                                fb = r["feedback_status"]
                                sub_topic = r["sub_topic"]
                                cos_m = r["cosine_score"]
                                r2_m = r["rouge2_score"]
                                rL_m = r["rougeL_score"]
                                srcs = r["source_docs"]
                                rel_th = r["relevance_threshold"]
                                k = r["top_k"]
                                header = f"Q: {q_text} (id: {rid}, created: {created_at}, sub_topic: {sub_topic}, feedback: {fb})"
                                with st.expander(header):
                                    st.write(a_text)
                                    # Display RAG Settings before Metrics
                                    st.markdown(f"**RAG Settings:** Relevance Threshold={rel_th}, top_k (Retrieval Count)={k}")
                                    st.markdown(f"**Metrics:** Cosine={cos_m}, ROUGE-2={r2_m}, ROUGE-L={rL_m}")
                                    st.markdown(f"**Source docs:** {srcs}")
                                    col1, col2, col3 = st.columns([1,1,4])
                                    with col1:
                                        # Disabled for viewer
                                        if st.button(f"Delete id {rid}", key=f"del_full_{rid}", disabled=is_viewer):
                                            delete_cache_entry(rid)
                                            st.success(f"Deleted cache entry id {rid}. Please reopen topic view to refresh.")
                                    with col2:
                                        # Disabled for viewer
                                        if st.button(f"Mark Helpful (Y) id {rid}", key=f"help_full_{rid}", disabled=is_viewer):
                                            update_feedback(rid, "Y")
                                            st.success(f"Marked id {rid} as Helpful (Y).")
                                    with col3:
                                        # Disabled for viewer
                                        if st.button(f"Mark Not Helpful (N) id {rid}", key=f"nothelp_full_{rid}", disabled=is_viewer):
                                            update_feedback(rid, "N")
                                            st.success(f"Marked id {rid} as Not Helpful (N).")

                if st.session_state.manage_filter_open.get(t, False):
                    with st.expander(f"Filter & Sort cached Q&A by metric values — Topic: {t}", expanded=True):
                        rows = fetch_cache_by_topic(t, only_helpful=False)
                        if not rows:
                            st.info("No cached Q&A for this topic.")
                        else:
                            df_rows = []
                            for row in rows:
                                rid, q_text, a_text, created_at, fb, sub_topic, cos_m, r2_m, rL_m, source_docs_list, rel_th, k = row
                                df_rows.append({
                                    "id": rid,
                                    "question": q_text,
                                    "answer": a_text,
                                    "created_at": created_at,
                                    "feedback_status": fb or "",
                                    "sub_topic": sub_topic or "all",
                                    "cosine_score": cos_m,
                                    "rouge2_score": r2_m,
                                    "rougeL_score": rL_m,
                                    "source_docs": json.dumps(source_docs_list),
                                    "relevance_threshold": rel_th,
                                    "top_k": k
                                })
                            df_cache = pd.DataFrame(df_rows)
                            st.markdown("**Filter controls**")
                            filter_cols = st.columns(4)
                            # Add unique keys using the topic variable 't'
                            enable_cos = filter_cols[0].checkbox("Filter Cosine", value=False, key=f"chk_cos_{t}")
                            enable_r2 = filter_cols[1].checkbox("Filter ROUGE-2", value=False, key=f"chk_r2_{t}")
                            enable_rL = filter_cols[2].checkbox("Filter ROUGE-L", value=False, key=f"chk_rL_{t}")
                                                        
                            # Use st.container() to prevent layout jump with dynamic sliders
                            with st.container():
                                if enable_cos:
                                    cos_min, cos_max = st.slider("Cosine range", 0.0, 1.0, (0.0,1.0), key=f"cos_range_{t}")
                                else:
                                    cos_min, cos_max = 0.0, 1.0
                                if enable_r2:
                                    r2_min, r2_max = st.slider("ROUGE-2 range", 0.0, 1.0, (0.0,1.0), key=f"r2_range_{t}")
                                else:
                                    r2_min, r2_max = 0.0, 1.0
                                if enable_rL:
                                    rL_min, rL_max = st.slider("ROUGE-L range", 0.0, 1.0, (0.0,1.0), key=f"rL_range_{t}")
                                else:
                                    rL_min, rL_max = 0.0, 1.0
                                
                            df_shown = df_cache[
                                (df_cache["cosine_score"].fillna(0.0) >= cos_min) & (df_cache["cosine_score"].fillna(0.0) <= cos_max) &
                                (df_cache["rouge2_score"].fillna(0.0) >= r2_min) & (df_cache["rouge2_score"].fillna(0.0) <= r2_max) &
                                (df_cache["rougeL_score"].fillna(0.0) >= rL_min) & (df_cache["rougeL_score"].fillna(0.0) <= rL_max)
                            ]
                            if df_shown.empty:
                                st.info("No cached records match the selected filter criteria.")
                            else:
                                df_display = df_shown.copy()
                                df_display["answer_excerpt"] = df_display["answer"].str.slice(0, 500).str.replace("\n", " ")
                                display_cols = ["id", "created_at", "sub_topic", "feedback_status", "cosine_score", "rouge2_score", "rougeL_score", "source_docs", "question", "answer_excerpt"]
                                row_count = len(df_display)
                                table_height = max(160, min(800, 40 * row_count))
                                st.dataframe(df_display[display_cols].reset_index(drop=True), use_container_width=True, height=table_height)
                            st.markdown("**Note:** This filtered panel is read-only (no delete or feedback buttons). Use 'Manage Q&A' to operate on individual entries.")
    
        st.markdown("---")
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("↺ Clear summarization cache"):
                st.cache_data.clear()
                st.session_state.pop("cache_hits", None)
                st.success("✓ Cleared in-memory cache for chunk summaries.")
        with col_clear2:
            # Disabled for viewer
            if st.button("↺ Clear cache DB (delete ALL Q&A)", disabled=is_viewer):
                clear_cache_db()
                st.success("✓ cache DB cleared (all Q&A removed).")

# Safe rerun trigger after callbacks
if st.session_state.get("should_rerun", False):
    st.session_state["should_rerun"] = False
    st.rerun()