from pathlib import Path
from .reader import Reader
from .chunking import Chunker
from src.models import Chunk
from src.retrieval.bm25 import BM25
from tqdm import tqdm

class Indexer:
    def __init__(self, project_root: Path, corpus: Path, max_chunk_size: int = 2000) -> None:
        self.project_root = project_root
        self.corpus = corpus
        self.max_chunk_size = max_chunk_size
        self.reader = Reader(project_root)
        self.chunker = Chunker()
        self.failed_read: dict[Path, str] = {}

    def build(self) -> BM25:
        self.failed_read.clear()
        chunks: list[Chunk] = []
        source_file = self.reader.read_directory(self.corpus, self.failed_read)
        for f in tqdm(source_file, desc="Reading and chunking", unit="file"):
            chunks.extend(self.chunker.chunk(f, len(chunks), self.max_chunk_size))
        bm25 = BM25(chunks)
        bm25.build()
        return bm25