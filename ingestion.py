import os
import pickle
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_redis import RedisVectorStore
from config import Config

class DocumentIngestion:
    def __init__(self, hf_embeddings, cache_manager):
        self.hf_embeddings = hf_embeddings
        self.cache = cache_manager

    def load_and_chunk(self):
        loader = DirectoryLoader(
            Config.PDF_DIR,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            doc.metadata = {
                "source": source,
                "filename": os.path.basename(source),
                "page": doc.metadata.get("page", 0)
            }
        splitter = SemanticChunker(
            self.hf_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def save_chunks(self, chunks):
        with open(Config.CHUNK_FILE, "wb") as f:
            pickle.dump(chunks, f)

    def load_chunks(self):
        with open(Config.CHUNK_FILE, "rb") as f:
            return pickle.load(f)

    def build_vector_store(self, chunks):
        for chunk in chunks:
            self.cache.get_cached_embedding(chunk.page_content)
        store = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=self.hf_embeddings,
            redis_url=Config.REDIS_URL,
            index_name=Config.INDEX_NAME
        )
        return store

    def load_vector_store(self):
        return RedisVectorStore.from_existing_index(
            embedding=self.hf_embeddings,
            index_name=Config.INDEX_NAME,
            redis_url=Config.REDIS_URL
        )

    def run(self):
        if self.cache.exists(Config.INDEX_FLAG):
            print("Loading existing index...")
            chunks = self.load_chunks()
        else:
            print("Running ingestion pipeline...")
            chunks = self.load_and_chunk()
            self.save_chunks(chunks)
            self.build_vector_store(chunks)
            self.cache.set_flag(Config.INDEX_FLAG)
        vector_store = self.load_vector_store()
        return chunks, vector_store