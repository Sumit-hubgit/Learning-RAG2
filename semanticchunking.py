from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader
)

from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

import numpy as np
import os

load_dotenv()

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD PDFs
# -----------------------------

directory_loader = DirectoryLoader(
    "data/pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

docs = directory_loader.load()

# -----------------------------
# SEMANTIC CHUNKING
# -----------------------------

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

texts = [doc.page_content for doc in chunks]

chunk_embeddings = embedding_model.encode(texts)

print(chunk_embeddings.shape)

# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.1
)

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------

def retrieve(query, top_k=3):

    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_chunks = []

    for idx in top_indices:
        retrieved_chunks.append({
            "content": texts[idx],
            "score": similarities[idx]
        })

    return retrieved_chunks

# -----------------------------
# RAG QA FUNCTION
# -----------------------------

def rag_qa(query):

    retrieved_docs = retrieve(query)

    context = "\n\n".join([
        doc["content"] for doc in retrieved_docs
    ])

    prompt = f"""
Use the provided context to answer the question and do not answer on your own pretrained knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content

# -----------------------------
# ASK QUESTION
# -----------------------------

query = "What is RAG?"

answer = rag_qa(query)

print("\nANSWER:\n")
print(answer)