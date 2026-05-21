from sentence_transformers import CrossEncoder
from config import Config

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(Config.RERANKER_MODEL)

    def rerank(self, query, docs, top_k=None):
        top_k = top_k or Config.TOP_K_RERANK
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]