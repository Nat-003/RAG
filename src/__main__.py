from fire import Fire
from . import cli
from  pathlib import Path
from src.dataset_io import load_json
from src.models import RagDataset
from typing import TypeVar
from pydantic import BaseModel
from src.indexing.chunking import Chunker


def main() -> None:
    """Test Python chunk extraction and source integrity."""
    path = Path("test.py")
    max_size = 500
    chunker = Chunker()
    spans = chunker.parse_file(path, max_size)

    with path.open("r", encoding="utf-8", newline="") as file:
        source_code = file.read()

    print(f"File: {path}")
    print(f"Source length: {len(source_code)}")
    print(f"Total chunks: {len(spans)}")

    previous_end = 0
    for i, (start, end) in enumerate(spans, start=1):
        chunk = source_code[start:end]
        valid_range = 0 <= start < end <= len(source_code)
        valid_size = len(chunk) <= max_size
        no_overlap = start >= previous_end

        print(f"\n{'=' * 60}")
        print(f"Chunk {i}: [{start}:{end}]")
        print(f"Size: {len(chunk)} / {max_size}")
        print(f"Valid range: {valid_range}")
        print(f"Valid size: {valid_size}")
        print(f"No overlap: {no_overlap}")
        print(f"{'-' * 60}")
        print(chunk)

        assert valid_range, f"Invalid range in chunk {i}"
        assert valid_size, f"Oversized chunk {i}"
        assert no_overlap, f"Overlapping chunk {i}"
        previous_end = end

    print("\nAll chunk integrity checks passed!")


if __name__ == "__main__":
    main()
