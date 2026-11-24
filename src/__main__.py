from fire import Fire
from . import cli
from  pathlib import Path
from src.dataset_io import load_json
from src.models import RagDataset
from typing import TypeVar
from pydantic import BaseModel
from src.indexing.chunking import Chunker

"""Smoke-test the Python SourceFile-to-Chunk pipeline."""

"""Inspect the existing Reader-to-Chunker pipeline on local sources."""

import argparse
from pathlib import Path

from src.indexing.chunking import Chunker
from src.indexing.reader import Reader
from src.models import Chunk, SourceFile


def check_chunks(
    source: SourceFile, chunks: list[Chunk], first_id: int, max_size: int
) -> None:
    """Verify chunk ID order, exact source offsets and meaningful coverage."""
    cursor = 0
    for index, chunk in enumerate(chunks):
        start = chunk.source.first_character_index
        end = chunk.source.last_character_index
        if chunk.chunk_id != first_id + index:
            raise ValueError("Non-sequential chunk IDs")
        if chunk.source.file_path != source.file_path:
            raise ValueError("Mismatched source file path")
        if not cursor <= start < end <= len(source.text):
            raise ValueError("Invalid or overlapping source coordinates")
        if chunk.text != source.text[start:end]:
            raise ValueError("Chunk text differs from original source")
        if not 0 < len(chunk.text) <= max_size or not chunk.text.strip():
            raise ValueError("Empty or oversized chunk")
        if source.text[cursor:start].strip():
            raise ValueError("Meaningful text lost before chunk")
        cursor = end
    if source.text[cursor:].strip():
        raise ValueError("Meaningful text lost after final chunk")


def main() -> None:
    """Read eligible files, chunk them and report verified statistics."""
    parser = argparse.ArgumentParser(description="Reader + Chunker smoke test")
    parser.add_argument("--corpus", type=Path, default=Path("."))
    parser.add_argument("--max-size", type=int, default=2000)
    args = parser.parse_args()

    reader = Reader(Path.cwd())
    chunker = Chunker()
    failed: dict[Path, str] = {}
    chunks: list[Chunk] = []
    file_count = 0
    file_types: dict[str, int] = {}

    for source in reader.read_directory(args.corpus, failed):
        new_chunks = chunker.chunk(
            source, first_id=len(chunks), max_size=args.max_size
        )
        check_chunks(source, new_chunks, len(chunks), args.max_size)
        file_count += 1
        file_types[source.file_type] = file_types.get(source.file_type, 0) + 1
        chunks.extend(new_chunks)

    print(f"Verified files: {file_count}")
    print(f"Types: {file_types}")
    print(f"Valid chunks: {len(chunks)}")
    print(f"Skipped files: {len(failed)}")
    for path, reason in failed.items():
        print(f"  {path}: {reason}")
    print("Reader → Chunker integrity checks passed")


if __name__ == "__main__":
    main()