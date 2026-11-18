from fire import Fire
from . import cli
from  pathlib import Path
from src.dataset_io import load_json
from src.models import RagDataset
from typing import TypeVar
from pydantic import BaseModel
from src.indexing.chunking import Chunker

"""Smoke-test the Python SourceFile-to-Chunk pipeline."""

from pathlib import Path

from src.indexing.chunking import Chunker
from src.models import Chunk, SourceFile


def read_source(path: Path) -> SourceFile:
    """Build a SourceFile from a local Python file without changing newlines."""
    with path.open("r", encoding="utf-8", newline="") as file:
        text = file.read()
    return SourceFile(file_path=path.as_posix(), text=text, file_type="python")


def validate_chunks(
    source: SourceFile, chunks: list[Chunk], first_id: int, max_size: int
) -> None:
    """Check chunk IDs, sizes, offsets, text, paths, and source coverage."""
    previous_end = 0
    covered: set[int] = set()
    for index, chunk in enumerate(chunks):
        start = chunk.source.first_character_index
        end = chunk.source.last_character_index
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_id == first_id + index
        assert chunk.source.file_path == source.file_path
        assert previous_end <= start < end <= len(source.text)
        assert 0 < len(chunk.text) <= max_size
        assert chunk.text == source.text[start:end]
        assert chunk.text.strip()
        covered.update(range(start, end))
        previous_end = end
    assert all(
        position in covered or character.isspace()
        for position, character in enumerate(source.text)
    ), f"Lost meaningful source text: {source.file_path}"


def main() -> None:
    """Exercise real files, Unicode, invalid syntax, and error handling."""
    chunker = Chunker()
    # with open("test.md", "r") as f:
    #     content = f.read()
    #     chunker.parse_plain_text(content, 1)
    spans = [
    (0, 300),
    (302, 700),
    (702, 1200),
    (1202, 2500)
]
    chunker.merge_paragraphs(spans, 1)
    # paths = [Path("src/indexing/chunking.py"), Path("src/models.py")]
    # reader_path = Path("src/indexing/reader.py")
    # if reader_path.is_file():
    #     paths.append(reader_path)
    # sources = [read_source(path) for path in paths]
    # sources.append(SourceFile(
    #     file_path="data/raw/unicode_sample.py",
    #     file_type="python",
    #     text=(
    #         "# Présentation\r\n"
    #         "@decorator\r\n"
    #         "def café():\r\n"
    #         "    def intérieur():\r\n"
    #         "        return '🍰'\r\n"
    #         "    return intérieur()\r\n"
    #         "\r\n"
    #         "# Commentaire final\r\n"
    #     ),
    # ))
    # sources.append(SourceFile(
    #     file_path="data/raw/invalid_sample.py",
    #     file_type="python",
    #     text="def broken(:\n    return 'not valid Python'\n",
    # ))
    # for max_size in (100, 2000):
    #     all_chunks: list[Chunk] = []
    #     for source in sources:
    #         first_id = len(all_chunks)
    #         chunks = chunker.chunk(source, first_id=first_id, max_size=max_size)
    #         validate_chunks(source, chunks, first_id, max_size)
    #         all_chunks.extend(chunks)
    #         print(f"PASS  size={max_size:4}  chunks={len(chunks):3}  {source.file_path}")
    #     assert len({chunk.chunk_id for chunk in all_chunks}) == len(all_chunks)
    #     print(f"PASS  unique IDs across {len(sources)} files: {len(all_chunks)} chunks")
    # for bad_size in (0, 2001):
    #     try:
    #         chunker.chunk(sources[0], max_size=bad_size)
    #     except ValueError:
    #         print(f"PASS  rejected max_size={bad_size}")
    #     else:
    #         raise AssertionError(f"Expected ValueError for max_size={bad_size}")
    # try:
    #     chunker.chunk(SourceFile(
    #         file_path="data/raw/readme.md", text="# Title", file_type="markdown"
    #     ))
    # except ValueError:
    #     print("PASS  Markdown rejected until its chunking strategy is implemented")
    # else:
    #     raise AssertionError("Expected unsupported Markdown to be rejected")
    # print("\nAll Python chunking smoke tests passed.")


if __name__ == "__main__":
    main()

