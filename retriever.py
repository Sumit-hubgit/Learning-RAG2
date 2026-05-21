from langchain_community.retrievers import BM25Retriever
from config import Config

class HybridRetriever:
    def __init__(self, chunks, vector_store):
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = Config.TOP_K_RETRIEVE
        self.vector = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.TOP_K_RETRIEVE}
        )

    def reciprocal_rank_fusion(self, bm25_docs, vector_docs, k=60):
        scores, doc_map = {}, {}
        for rank, doc in enumerate(bm25_docs):
            doc_map[doc.page_content] = doc
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(vector_docs):
            doc_map[doc.page_content] = doc
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in ranked]

    def retrieve(self, query):
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vector.invoke(query)
        return self.reciprocal_rank_fusion(bm25_docs, vector_docs)