from src.models import Chunk
from .tokenizer import tokenize
from collections import Counter

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

