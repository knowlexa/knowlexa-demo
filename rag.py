import faiss
import numpy as np
import pickle
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
META_FILE = os.path.join(DATA_DIR, "metadata.pkl")


def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def split_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks


def save_index(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)


def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []


def reset_index():
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(META_FILE):
        os.remove(META_FILE)


def add_documents(files, index=None, metadata=None):
    if metadata is None:
        metadata = []

    new_chunks = []
    for file in files:
        text = load_pdf(file)
        chunks = split_text(text)
        for c in chunks:
            metadata.append({
                "text": c,
                "source": file.name
            })
            new_chunks.append(c)

    embeddings = embedder.encode(new_chunks).astype("float32")

    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    save_index(index, metadata)

    return index, metadata


def ask_question(question, index, metadata):
    q_embedding = embedder.encode([question]).astype("float32")
    D, I = index.search(q_embedding, k=3)

    retrieved = [metadata[i] for i in I[0]]

    context = "\n\n".join([r["text"] for r in retrieved])
    sources = list(set([r["source"] for r in retrieved]))

    prompt = f"""
Context:
{context}

Question:
{question}

Answer using only the context above.
If not found, say "Not available in the document."
"""

    result = qa_model(prompt, max_new_tokens=200)
    answer = result[0]["generated_text"].strip()


    return answer, sources, context
