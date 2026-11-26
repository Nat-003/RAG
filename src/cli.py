from pathlib import Path
from src.indexing.indexer import Indexer
from src.storage.index_store import IndexStorage
import sys

def index(
    corpus_path: str = "data/raw",
    storage_path: str = "data/processed",
    max_chunk_size: int = 2000,
) -> None:
    """Build and persist the corpus index."""
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("Invalid chunk size")
    root = Path.cwd()
    corpus_file_path = Path(corpus_path)
    storage_file_path = Path(storage_path)
    indexer = Indexer(root, corpus_file_path, max_chunk_size)
    bm25 = indexer.build()
    if indexer.failed_read:
        print(f"Read issues: {len(indexer.failed_read)}", file=sys.stderr)
        for path, reason in indexer.failed_read.items():
            print(f"  {path}: {reason}", file=sys.stderr)
    index_storage = IndexStorage(storage_file_path)
    index_storage.save(bm25, max_chunk_size)
    print(f"Indexed {bm25.total_documents} chunks")
    print(f"Index saved to {storage_file_path}")

def search(query: str, k: int = 5) -> None:
    print(f"Searching for {query}\nTop k: {k}")

def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    print(f"Dataset path: {dataset_path}\nTop K: {k}\n save_directory: {save_directory}")

def answer(query: str, k: int = 5) -> None:
    print(f"answer for query: {query}\nTop K: {k}")

def answer_dataset(student_search_results_path: str, save_directory: str) -> None:
    print(f"student result path: {student_search_results_path}\nSave directory: {save_directory}")

def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    print(f"student result path: {student_search_results_path}\nDataset path: {dataset_path}")
    
