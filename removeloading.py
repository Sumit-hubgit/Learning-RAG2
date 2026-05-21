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
import pickle
import os

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

# -----------------------------
# REDIS CONNECTION
# -----------------------------

redis_client = redis.Redis(
    host="127.0.0.1",
    port=6379,
    decode_responses=False
)

print("Connected to Redis")

# -----------------------------
# CONSTANTS
# -----------------------------

INDEX_NAME = "advanced-rag-index"

INDEX_FLAG = "advanced_rag_ready"

CHUNK_FILE = "chunks.pkl"

REDIS_URL = "redis://127.0.0.1:6379"

# -----------------------------
# EMBEDDING MODELS
# -----------------------------

hf_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# RERANKER MODEL
# -----------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

    embedding = embedding_model.encode(text)

    try:

        redis_client.set(
            key,
            embedding.astype(np.float32).tobytes()
        )

    except Exception as e:

        print(f"Redis embedding cache failed: {e}")

    print("Embedding stored in Redis")

    return embedding

# -----------------------------
# RESPONSE CACHE
# -----------------------------

def get_cached_response(query):

    key = f"response:{hashlib.md5(query.encode()).hexdigest()}"

    try:

        cached_response = redis_client.get(key)

        if cached_response:

            print("LLM response fetched from Redis cache")

            return cached_response.decode()

    except Exception as e:

        print(f"Redis response cache read failed: {e}")

    return None


def cache_response(query, response):

    key = f"response:{hashlib.md5(query.encode()).hexdigest()}"

    try:

        redis_client.setex(
            key,
            3600,
            response
        )

    except Exception as e:

        print(f"Redis response cache write failed: {e}")

# -----------------------------
# PREPROCESSING
# -----------------------------

if redis_client.exists(INDEX_FLAG):

    print("\nPreprocessed data already exists")

    with open(CHUNK_FILE, "rb") as f:

        chunks = pickle.load(f)

    print(f"\nLoaded {len(chunks)} chunks from disk")

else:

    print("\nRunning preprocessing pipeline...\n")

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

    # -----------------------------
    # ADD METADATA
    # -----------------------------

    for doc in documents:

        source_path = doc.metadata.get(
            "source",
            "unknown"
        )

        filename = os.path.basename(source_path)

        page_number = doc.metadata.get(
            "page",
            0
        )

        doc.metadata = {
            "source": source_path,
            "filename": filename,
            "page": page_number
        }

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
    # SAVE CHUNKS
    # -----------------------------

    with open(CHUNK_FILE, "wb") as f:

        pickle.dump(chunks, f)

    print("\nChunks saved to disk")

    # -----------------------------
    # CACHE EMBEDDINGS
    # -----------------------------

    print("\nCaching embeddings...\n")

    for chunk in chunks:

        get_cached_embedding(
            chunk.page_content
        )

    print("\nAll embeddings cached")

    # -----------------------------
    # CREATE VECTOR STORE
    # -----------------------------

    vector_store = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=hf_embeddings,
        redis_url=REDIS_URL,
        index_name=INDEX_NAME
    )

    print("\nRedis vector index created")

    # -----------------------------
    # SET REDIS FLAG
    # -----------------------------

    redis_client.set(
        INDEX_FLAG,
        "true"
    )

# -----------------------------
# BM25 RETRIEVER
# -----------------------------

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 10

print("BM25 Retriever Initialized")

# -----------------------------
# LOAD VECTOR STORE
# -----------------------------

vector_store = RedisVectorStore.from_existing_index(
    embedding=hf_embeddings,
    index_name=INDEX_NAME,
    redis_url=REDIS_URL
)

print("\nLoaded existing Redis vector index")

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
# RRF
# -----------------------------

def reciprocal_rank_fusion(
    bm25_docs,
    vector_docs,
    k=60
):

    scores = {}

    doc_map = {}

    for rank, doc in enumerate(bm25_docs):

        content = doc.page_content

        doc_map[content] = doc

        scores[content] = scores.get(content, 0) + (
            1 / (k + rank + 1)
        )

    for rank, doc in enumerate(vector_docs):

        content = doc.page_content

        doc_map[content] = doc

        scores[content] = scores.get(content, 0) + (
            1 / (k + rank + 1)
        )

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
# HYBRID RETRIEVAL
# -----------------------------

def hybrid_retrieve(query):

    print(f"\nSearching for query: {query}")

    bm25_docs = bm25_retriever.invoke(query)

    vector_docs = vector_retriever.invoke(query)

    fused_docs = reciprocal_rank_fusion(
        bm25_docs,
        vector_docs
    )

    print(f"\nRRF Retrieved Docs: {len(fused_docs)}")

    return fused_docs

# -----------------------------
# RERANKING
# -----------------------------

def rerank_documents(
    query,
    docs,
    top_k=5
):

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(
        zip(docs, scores)
    )

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = scored_docs[:top_k]

    print("\nTop Reranked Chunks:\n")

    final_docs = []

    for i, (doc, score) in enumerate(
        reranked_docs,
        1
    ):

        print(f"\nRank {i} | Score: {score:.4f}")

        print(
            f"Source: {doc.metadata.get('filename')}"
        )

        print(
            f"Page: {doc.metadata.get('page')}"
        )

        print("-" * 80)

        print(doc.page_content[:700])

        print("=" * 100)

        final_docs.append(doc)

    return final_docs

# -----------------------------
# RAG QA
# -----------------------------

def rag_qa(query):

    cached_response = get_cached_response(query)

    if cached_response:

        return cached_response

    retrieved_docs = hybrid_retrieve(query)

    reranked_docs = rerank_documents(
        query,
        retrieved_docs,
        top_k=5
    )

    context = "\n\n".join([
        doc.page_content
        for doc in reranked_docs
    ])

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not present,
say:
"I could not find the answer in the provided documents."

Give answers in detailed bullet points.

Context:
{context}

Question:
{query}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        final_answer = response.content

    except Exception as e:

        return f"LLM Error: {str(e)}"

    cache_response(
        query,
        final_answer
    )

    return final_answer

# -----------------------------
# ASK QUESTION
# -----------------------------

query = "Explain the transformer model architecture"

answer = rag_qa(query)

print("\nFINAL ANSWER:\n")

print(answer)