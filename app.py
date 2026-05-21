from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

from config import Config
from cache import CacheManager
from ingestion import DocumentIngestion
from retriever import HybridRetriever
from reranker import Reranker

class RAGPipeline:
    def __init__(self):
        hf_embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        self.cache = CacheManager(embedding_model)
        ingestion = DocumentIngestion(hf_embeddings, self.cache)
        chunks, vector_store = ingestion.run()

        self.retriever = HybridRetriever(chunks, vector_store)
        self.reranker = Reranker()
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model=Config.LLM_MODEL,
            temperature=0.1,
            max_tokens=1024
        )

    def answer(self, query):
        cached = self.cache.get_cached_response(query)
        if cached:
            return cached

        docs = self.retriever.retrieve(query)
        docs = self.reranker.rerank(query, docs)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""You are a helpful AI assistant.
Answer ONLY from the provided context.
If the answer is not present, say: "I could not find the answer in the provided documents."
Give answers in summary bullet points.

Context:
{context}

Question:
{query}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            return f"LLM Error: {str(e)}"

        self.cache.cache_response(query, answer)
        return answer


if __name__ == "__main__":
    pipeline = RAGPipeline()
    query = "Explain the Attention Visualizations"
    print(pipeline.answer(query))