# 📚 Advanced-RAG-Pipeline

A Advanced Retrieval-Augmented Generation (RAG) pipeline built using LangChain, Redis Vector Store, Sentence Transformers, BM25, Semantic Search, and Groq LLMs.
This project demonstrates how Large Language Models can answer questions from custom PDF documents using semantic search and vector embeddings.

---

# 🚀 Features

* PDF document loading
* Intelligent text chunking
* Sentence Transformer embeddings
* Vector storage using ChromaDB
* Semantic similarity search
* Context-aware question answering
* Groq LLM integration
* End-to-end RAG pipeline implementation

---

# 🛠️ Tech Stack

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming language |
| LangChain             | RAG orchestration         |
| ChromaDB              | Vector database           |
| Sentence Transformers | Embedding generation      |
| MiniLM-L6-v2          | Embedding model           |
| Groq API              | LLM inference             |
| Llama 3.1 8B Instant  | Language model            |
| PyMuPDF               | PDF loading               |
| NumPy                 | Numerical operations      |

---

# 📂 Project Structure

```bash
Learning-RAG/
│
├── data/
│   ├── pdf/
│   │   └── sample.pdf
│   │
│   └── vector_store/
│
├── ragpipeline.py
├── requirements.txt
├── .env
└── README.md
```

---

# ⚙️ How It Works

The project follows a complete RAG workflow:

1. Load PDF documents
2. Split documents into smaller chunks
3. Generate embeddings for each chunk
4. Store embeddings in ChromaDB
5. Convert user query into embeddings
6. Retrieve semantically similar chunks
7. Send retrieved context to the LLM
8. Generate final answer

---

# 🧠 Core Logic

## 1. Document Loading

PDF files are loaded using `PyMuPDFLoader`.

```python
dir_loader = DirectoryLoader(
    "data/pdf",
    glob ="**/*.pdf",
    loader_cls=PyMuPDFLoader
)
```

---

## 2. Text Chunking

Documents are divided into smaller chunks for efficient retrieval.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

---

## 3. Embedding Generation

Each chunk is converted into vector embeddings using Sentence Transformers.

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

---

## 4. Vector Storage

Embeddings are stored inside ChromaDB for semantic search.

```python
chromadb.PersistentClient()
```

---

## 5. Semantic Retrieval

User queries are embedded and compared against stored embeddings to retrieve relevant context.

---

## 6. LLM Response Generation

Retrieved chunks are passed to Groq-hosted Llama 3.1 model to generate answers.

```python
ChatGroq(
    model="llama-3.1-8b-instant"
)
```

---

# 📌 Example Query

```python
answer = rag_qa(
    "Who wrote Attention is all you need",
    retriever,
    llm
)
```

---

# ✅ Example Output

```text
The paper "Attention Is All You Need" was written by
Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin.
```

---

# 💡 Where This Can Be Used

This RAG pipeline can be extended for:

* AI Chatbots
* Company Knowledge Bases
* PDF Question Answering Systems
* Legal Document Search
* Research Paper Assistants
* Customer Support Systems
* Resume Screening
* Medical Information Retrieval
* Enterprise Search Engines

---

# 🎯 Outcome of the Project

By building this project, you learn:

* How RAG systems work internally
* Semantic search concepts
* Vector databases
* Embedding generation
* Context injection into LLMs
* End-to-end AI pipeline development
* Practical LangChain integration
* Real-world LLM application design

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

# ▶️ Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python ragpipeline.py
```

---

# 🔄 Advanced RAG Pipeline Architecture

```text
                    +----------------------+
                    |    PDF Documents     |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |   Document Loader    |
                    | PyMuPDF + LangChain  |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |  Metadata Addition   |
                    | filename/page/source |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |  Semantic Chunking   |
                    |   SemanticChunker    |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |    Chunk Storage     |
                    |     chunks.pkl       |
                    +----------------------+
                                |
              +-----------------+------------------+
              |                                    |
              v                                    v
 +--------------------------+       +--------------------------+
 |  Embedding Generation    |       |     BM25 Retriever      |
 | SentenceTransformer      |       |   Sparse Retrieval       |
 +--------------------------+       +--------------------------+
              |                                    
              v                                    
 +--------------------------+       
 | Redis Embedding Cache    |
 +--------------------------+
              |
              v
 +--------------------------+
 | Redis Vector Database    |
 | HNSW ANN Vector Index    |
 +--------------------------+
              |
              v
 +--------------------------+
 | Semantic Vector Search   |
 +--------------------------+

========================================================
                QUERY RETRIEVAL PIPELINE
========================================================

                    +----------------------+
                    |      User Query      |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |    Hybrid Retrieval  |
                    | BM25 + Vector Search |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |   RRF Fusion Layer   |
                    | Reciprocal Rank      |
                    | Fusion               |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    | Cross-Encoder        |
                    | Reranking            |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    | Retrieved Context    |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    |  Groq LLM (Llama3.1) |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    | Response Cache       |
                    | Redis TTL Cache      |
                    +----------------------+
                                |
                                v
                    +----------------------+
                    | Final Generated      |
                    | Answer               |
                    +----------------------+
```
---

# 📖 Future Improvements

* Hybrid Search
* Metadata Filtering
* Streaming Responses
* Reranking
* Multi-document Support
* Conversational Memory
* MMR Retrieval
* Dockerization

---

# 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and improve the pipeline.

---

# ⭐ Acknowledgements

* LangChain
* ChromaDB
* HuggingFace
* Groq
* Sentence Transformers

---
