from pathlib import Path
from src.models import Chunk
from retrieval.bm25 import BM25
from src.models import Chunk, BM25State, IndexManifest
import json


class IndexStorage:
    """Manage the persistence of the RAG index."""
    def __init__(self, storage_path: Path) -> None:
        """Initialize the storage directory."""
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save chunks in JSONL format."""
        file_path = self.storage_path / "chunks.jsonl"
        with file_path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(chunk.model_dump_json() + "\n")

    def save_bm25(self, bm25: BM25) -> None:
        """Save BM25 index and corpus statistics in JSON format."""
        file_path = self.storage_path / "bm25_index.json"
        stats = {
            "inverted_index": bm25.inverted_index,
            "document_lengths": bm25.document_lengths,
            "total_documents": bm25.total_documents,
            "avgdl": bm25.avgdl,
        }
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(stats, file)

    def save_manifest(self, bm25: BM25, max_chunk_size: int) -> None:
        """Save index configuration and compatibility versions."""
        file_path = self.storage_path / "manifest.json"
        manifest = {
            "format_version": 1,
            "tokenizer_version": 1,
            "k1": bm25.k1,
            "b": bm25.b,
            "max_chunk_size": max_chunk_size,
        }
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4)

    def save(self, bm25: BM25, max_chunk_size: int) -> None:
        self.save_chunks(bm25.chunks)
        self.save_bm25(bm25)
        self.save_manifest(bm25, max_chunk_size)

    def load_chunks(self) -> list[Chunk]:
        """Load chunks from the JSONL file."""
        file_path = self.storage_path / "chunks.jsonl"
        chunks: list[Chunk] = []
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                chunks.append(Chunk.model_validate_json(line))
        return chunks

    def load_bm25(self) -> BM25State:
        """Load and validate persisted BM25 statistics."""
        file_path = self.storage_path / "bm25_index.json"
        return BM25State.model_validate_json(file_path.read_bytes())

    def load_manifest(self) -> IndexManifest:
        """Load and validate the index configuration."""
        file_path = self.storage_path / "manifest.json"
        return IndexManifest.model_validate_json(file_path.read_bytes())

    def load(self) -> BM25:
        """Load and restore a persisted BM25 index."""
        manifest = self.load_manifest()
        chunks = self.load_chunks()
        bm_stats = self.load_bm25()
        if len(chunks) != bm_stats.total_documents:
            raise ValueError("Inconsistent number of chunks")
        if len(bm_stats.document_lengths) != bm_stats.total_documents:
            raise ValueError("Inconsistent document lengths")
        bm25 = BM25(chunks)
        bm25.document_lengths = bm_stats.document_lengths
        bm25.inverted_index = bm_stats.inverted_index
        bm25.total_documents = bm_stats.total_documents
        bm25.avgdl = bm_stats.avgdl
        bm25.k1 = manifest.k1
        bm25.b = manifest.b
        return bm25