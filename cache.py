import redis
import hashlib
import numpy as np
from config import Config

class CacheManager:
    def __init__(self, embedding_model):
        self.client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=False
        )
        self.embedding_model = embedding_model

    def get_cached_embedding(self, text):
        key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.client.get(key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)
        embedding = self.embedding_model.encode(text)
        try:
            self.client.set(key, embedding.astype(np.float32).tobytes())
        except Exception as e:
            print(f"Embedding cache write failed: {e}")
        return embedding

    def get_cached_response(self, query):
        key = f"response:{hashlib.md5(query.encode()).hexdigest()}"
        try:
            cached = self.client.get(key)
            if cached:
                return cached.decode()
        except Exception as e:
            print(f"Response cache read failed: {e}")
        return None

    def cache_response(self, query, response):
        key = f"response:{hashlib.md5(query.encode()).hexdigest()}"
        try:
            self.client.setex(key, Config.CACHE_TTL, response)
        except Exception as e:
            print(f"Response cache write failed: {e}")

    def exists(self, flag):
        return self.client.exists(flag)

    def set_flag(self, flag):
        self.client.set(flag, "true")