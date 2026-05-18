from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

import numpy as np
import os

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

# -----------------------------
# EMBEDDING MODELS
# -----------------------------

# Used for semantic chunking
hf_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Used for embedding generation
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD PDF DOCUMENTS
# -----------------------------

directory_loader = DirectoryLoader(
    "data/pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

documents = directory_loader.load()

print(f"\nLoaded {len(documents)} PDF documents")

# -----------------------------
# SEMANTIC CHUNKING
# -----------------------------

text_splitter = SemanticChunker(
    hf_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = text_splitter.split_documents(documents)

print(f"\nTotal semantic chunks created: {len(chunks)}")

# -----------------------------
# EXTRACT TEXT FROM CHUNKS
# -----------------------------

texts = [doc.page_content for doc in chunks]

# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------

print("\nGenerating embeddings...\n")

chunk_embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print(f"\nEmbedding Shape: {chunk_embeddings.shape}")

# -----------------------------
# INITIALIZE LLM
# -----------------------------

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------

def retrieve(query, top_k=5):

    print(f"\nSearching for query: {query}")

    # Convert query into embedding
    query_embedding = embedding_model.encode([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    # Get top-k highest similarity chunks
    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_chunks = []

    print("\nTop Retrieved Chunks:\n")

    for idx in top_indices:

        retrieved_chunks.append({
            "content": texts[idx],
            "score": similarities[idx]
        })

        print(f"Similarity Score: {similarities[idx]:.4f}")
        print("-" * 80)
        print(texts[idx][:700])
        print("=" * 100)

    return retrieved_chunks

# -----------------------------
# RAG QUESTION ANSWERING
# -----------------------------

def rag_qa(query):

    # Retrieve relevant chunks
    retrieved_docs = retrieve(query)

    # Combine context
    context = "\n\n".join([
        doc["content"] for doc in retrieved_docs
    ])

    # Prompt Template
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If exact names or details exist in the context,
return them completely without summarizing.

If the answer is not present in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate response
    response = llm.invoke(prompt)

    return response.content

# -----------------------------
# ASK QUESTIONS
# -----------------------------

query = "Tell me about Advanced RAG. Tell me the answers in point not in summary"

answer = rag_qa(query)

print("\nFINAL ANSWER:\n")
print(answer)

# FINAL ANSWER:
# 1. The development of Advanced RAG is a response to the specific shortcomings in Naive RAG.
# 2. Advanced RAG is a stage in the RAG research paradigm, which is continuously evolving.
# 3. The RAG research paradigm is categorized into three stages: Naive RAG, Advanced RAG, and Modular RAG.
# 4. Figure 3 shows the three stages of the RAG research paradigm, including Advanced RAG.
# 5. Advanced RAG is a response to the limitations o