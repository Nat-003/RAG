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

from fire import Fire
from . import cli
import sys


def main() -> None:
    """Launch the RAG CLI and handle application errors."""
    try:
        Fire({
            "index": cli.index,
            "search": cli.search,
            "search_dataset": cli.search_dataset,
            "answer": cli.answer,
            "answer_dataset": cli.answer_dataset,
            "evaluate": cli.evaluate,
        })
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()