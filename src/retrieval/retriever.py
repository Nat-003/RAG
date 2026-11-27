from .bm25 import BM25
from src.models import Chunk, RetrievedChunk

class Retriever:
    def __init__(self, bm25: BM25) -> None:
        self.bm25 = bm25
        self.lookup_chunk: dict[int, Chunk] = {}
        for chunk in self.bm25.chunks:
            chunk_id = chunk.chunk_id
            self.lookup_chunk[chunk_id] = chunk

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k retrieved chunks with their BM25 scores."""
        search_res = self.bm25.search(query, k)
        result: list[RetrievedChunk] = []
        for chunk_id, score in search_res:
            if chunk_id not in self.lookup_chunk:
                raise ValueError(f"Missing chunk ID: {chunk_id}")
            chunk = self.lookup_chunk[chunk_id]
            retrieved_chunk = RetrievedChunk(chunk=chunk, score=score)
            result.append(retrieved_chunk)
        return result