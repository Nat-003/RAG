from src.models import Chunk
from .tokenizer import tokenize
from collections import Counter
import math

class BM25:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.inverted_index: dict[str, dict[int, int]] = {}
        self.document_lengths: 	dict[int, int] = {}
        self.total_documents: int = 0
        self.avgdl: float = 0.0
        self.k1: float = 1.5
        self.b: float = 0.75

    
    def build(self) -> None:
        """Build the BM25 inverted index and document statistics."""
        if not self.chunks:
            raise ValueError("No chunks to process")
        self.inverted_index = {}
        self.document_lengths = {}
        self.total_documents = 0
        self.avgdl = 0.0
        total_tokens = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            frequencies = Counter(tokens)
            for term, frequency in frequencies.items():
                self.inverted_index.setdefault(term, {})[chunk.chunk_id] = frequency
            self.document_lengths[chunk.chunk_id] = len(tokens)
            self.total_documents += 1
            total_tokens += len(tokens)
        self.avgdl = total_tokens / self.total_documents

    def calculate_idf(self, term: str) -> float:
        """Calculate the inverse document frequency of a term."""
        df = len(self.inverted_index[term])
        idf = math.log(1 + (self.total_documents - df + 0.5) / (df + 0.5))
        return idf

    def calculate_score(self, term: str, chunk_id: int) -> float:
        """Calculate the BM25 score of a term within a chunk."""
        idf = self.calculate_idf(term)
        d = self.document_lengths[chunk_id]
        tf = self.inverted_index[term][chunk_id]
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (d / self.avgdl))
        return idf * (numerator / denominator)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the top-k chunks ranked by BM25 score."""
        if k <= 0 or self.avgdl <= 0:
            return []
        tokens = tokenize(query)
        unique_tokens = dict.fromkeys(tokens)
        chunk_score: dict[int, float] = {}
        for token in unique_tokens:
            if token not in self.inverted_index:
                continue
            for chunk_id in self.inverted_index[token]:
                score = self.calculate_score(token, chunk_id)
                chunk_score[chunk_id] = chunk_score.get(chunk_id, 0.0) + score
        ranked = sorted(chunk_score.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
            
