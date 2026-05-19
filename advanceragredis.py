from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_redis import RedisVectorStore
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from dotenv import load_dotenv

import redis
import hashlib
import numpy as np
import os

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

# -----------------------------
# REDIS CONNECTION
# -----------------------------

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=False
)

print("Connected to Redis")

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
# RERANKER MODEL
# -----------------------------

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker model loaded")

# -----------------------------
# EMBEDDING CACHE FUNCTION
# -----------------------------

def get_cached_embedding(text):

    key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"

    cached_embedding = redis_client.get(key)

    if cached_embedding:

        print("Embedding fetched from Redis cache")

        return np.frombuffer(
            cached_embedding,
            dtype=np.float32
        )

    # Generate embedding
    embedding = embedding_model.encode(text)

    # Store embedding in Redis
    redis_client.set(
        key,
        embedding.astype(np.float32).tobytes()
    )

    print("Embedding stored in Redis")

    return embedding

# -----------------------------
# RESPONSE CACHE FUNCTIONS
# -----------------------------

def get_cached_response(query):

    key = f"response:{hashlib.md5(query.encode()).hexdigest()}"

    cached_response = redis_client.get(key)

    if cached_response:

        print("LLM response fetched from Redis cache")

        return cached_response.decode()

    return None


def cache_response(query, response):

    key = f"response:{hashlib.md5(query.encode()).hexdigest()}"

    # Cache for 1 hour
    redis_client.setex(
        key,
        3600,
        response
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
# BM25 RETRIEVER
# -----------------------------

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 10

print("BM25 Retriever Initialized")

# -----------------------------
# PRELOAD EMBEDDINGS INTO CACHE
# -----------------------------

print("\nCaching embeddings...\n")

for chunk in chunks:
    get_cached_embedding(chunk.page_content)

print("\nAll embeddings cached successfully")

# -----------------------------
# CREATE REDIS VECTOR STORE
# -----------------------------

INDEX_NAME = "advanced-rag-index"

try:

    vector_store = RedisVectorStore.from_existing_index(
        embedding=hf_embeddings,
        index_name=INDEX_NAME,
        redis_url="redis://localhost:6379"
    )

    print("\nLoaded existing Redis vector index")

except Exception:

    print("\nCreating new Redis vector index...")

    vector_store = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=hf_embeddings,
        redis_url="redis://localhost:6379",
        index_name=INDEX_NAME
    )

    print("\nDocuments stored in Redis Vector DB")


# -----------------------------
# VECTOR RETRIEVER
# -----------------------------

vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)


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

# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------

def hybrid_retrieve(query):

    print(f"\nSearching for query: {query}")

    # Cache query embedding
    get_cached_embedding(query)

    # BM25 Retrieval
    bm25_docs = bm25_retriever.invoke(query)

    # Vector Retrieval
    vector_docs = vector_retriever.invoke(query)

    # Merge Results
    # Apply RRF Fusion
    fused_docs = reciprocal_rank_fusion(
        bm25_docs,
        vector_docs
    )

    print(f"\nRRF Retrieved Docs: {len(fused_docs)}")

    return fused_docs
# -----------------------------
# RECIPROCAL RANK FUSION (RRF)
# -----------------------------

def reciprocal_rank_fusion(
    bm25_docs,
    vector_docs,
    k=60
):

    scores = {}

    doc_map = {}

    # BM25 ranking scores
    for rank, doc in enumerate(bm25_docs):

        content = doc.page_content

        doc_map[content] = doc

        scores[content] = scores.get(content, 0) + (
            1 / (k + rank + 1)
        )

    # Vector ranking scores
    for rank, doc in enumerate(vector_docs):

        content = doc.page_content

        doc_map[content] = doc

        scores[content] = scores.get(content, 0) + (
            1 / (k + rank + 1)
        )

    # Sort by fused score
    ranked_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    fused_docs = [
        doc_map[content]
        for content, score in ranked_docs
    ]

    return fused_docs

# -----------------------------
# RERANKING
# -----------------------------

def rerank_documents(query, docs, top_k=5):

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = scored_docs[:top_k]

    print("\nTop Reranked Chunks:\n")

    final_docs = []

    for i, (doc, score) in enumerate(reranked_docs, 1):

        print(f"\nRank {i} | Score: {score:.4f}")

        print("-" * 80)

        print(doc.page_content[:700])

        print("=" * 100)

        final_docs.append(doc)

    return final_docs

# -----------------------------
# RAG QUESTION ANSWERING
# -----------------------------

def rag_qa(query):

    # Check cached LLM response
    cached_response = get_cached_response(query)

    if cached_response:

        return cached_response

    # Retrieve relevant chunks
    # Hybrid Retrieval
    retrieved_docs = hybrid_retrieve(query)

    # Reranking
    reranked_docs = rerank_documents(
        query,
        retrieved_docs,
        top_k=5
    )

    # Combine context
    context = "\n\n".join([
    doc.page_content
    for doc in reranked_docs
    ])

    # Prompt Template
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.



If the answer is not present in the context,
say:
"I could not find the answer in the provided documents."

Give answers in detailed bullet points.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate response
    response = llm.invoke(prompt)

    final_answer = response.content

    # Cache response
    cache_response(query, final_answer)

    return final_answer

# -----------------------------
# ASK QUESTIONS
# -----------------------------

query = "Who are the authors of Attention is all you need research Paper? "

answer = rag_qa(query)

print("\nFINAL ANSWER:\n")
print(answer)