"""Check Reader-to-Chunker integration against real file contents."""

from pathlib import Path

import pytest

from src.indexing.chunking import Chunker
from src.indexing.reader import Reader
from src.models import Chunk, SourceFile


def validate_source_chunks(
    source: SourceFile, chunks: list[Chunk], start_id: int, max_size: int
) -> None:
    """Check identifiers, paths, offsets and source coverage."""
    cursor = 0
    for index, chunk in enumerate(chunks):
        begin = chunk.source.first_character_index
        end = chunk.source.last_character_index
        assert chunk.chunk_id == start_id + index
        assert chunk.source.file_path == source.file_path
        assert cursor <= begin < end <= len(source.text)
        assert chunk.text == source.text[begin:end]
        assert 0 < len(chunk.text) <= max_size
        assert chunk.text.strip()
        assert not source.text[cursor:begin].strip()
        cursor = end
    assert not source.text[cursor:].strip()


def test_reader_chunker_all_types_and_global_ids(tmp_path: Path) -> None:
    """Process Python, Markdown and text using the actual Reader API."""
    corpus = tmp_path / "data" / "raw" / "vllm-example"
    docs = corpus / "docs"
    code = corpus / "vllm"
    docs.mkdir(parents=True)
    code.mkdir()
    (code / "sample.py").write_bytes(
        "# café\r\ndef hello():\r\n    return '🎉'\r\n".encode("utf-8")
    )
    (docs / "guide.md").write_bytes(
        ("# Guide\r\n\r\n" + "hello 🦊 " * 300 + "\r\n\r\nEnd\r\n").encode("utf-8")
    )
    (docs / "notes.txt").write_bytes(b"First paragraph.\n\nSecond paragraph.\n")
    (docs / "ignore.json").write_text("{}", encoding="utf-8")
    reader = Reader(tmp_path)
    chunker = Chunker()
    failed: dict[Path, str] = {}
    all_chunks: list[Chunk] = []
    seen_types: set[str] = set()
    for source in reader.read_directory(corpus, failed):
        first_id = len(all_chunks)
        chunks = chunker.chunk(source, first_id=first_id, max_size=120)
        validate_source_chunks(source, chunks, first_id, 120)
        assert source.file_path.startswith("data/raw/vllm-example/")
        seen_types.add(source.file_type)
        all_chunks.extend(chunks)
    assert not failed
    assert seen_types == {"python", "markdown", "text"}
    assert len({chunk.chunk_id for chunk in all_chunks}) == len(all_chunks)
    assert [chunk.chunk_id for chunk in all_chunks] == list(range(len(all_chunks)))
    assert all(chunk.source.file_path.endswith((".py", ".md", ".txt")) for chunk in all_chunks)


def test_read_file_preserves_crlf_and_unicode(tmp_path: Path) -> None:
    """Read byte-preserving line terminators as original character offsets."""
    p = tmp_path / "utf8.md"
    original = "# Été\r\n\r\nfrançais 🧩\r\n"
    p.write_bytes(original.encode("utf-8"))
    source = Reader(tmp_path).read_file(p)
    assert source.text == original
    assert source.file_path == "utf8.md"
    chunks = Chunker().chunk(source, first_id=8, max_size=30)
    validate_source_chunks(source, chunks, 8, 30)


def test_reader_skips_invalid_utf8_and_empty_files(tmp_path: Path) -> None:
    """Record bad input files while continuing with valid ones."""
    corpus = tmp_path / "data" / "raw"
    corpus.mkdir(parents=True)
    (corpus / "bad.txt").write_bytes(b"\xff\xfe")
    (corpus / "empty.md").write_bytes(b"")
    (corpus / "valid.py").write_text("x = 42\n", encoding="utf-8")
    failed: dict[Path, str] = {}
    result = list(Reader(tmp_path).read_directory(corpus, failed))
    assert [s.file_path for s in result] == ["data/raw/valid.py"]
    assert len(failed) == 2
    assert len(Chunker().chunk(result[0])) == 1


def test_invalid_project_path_is_rejected(tmp_path: Path) -> None:
    """The Reader must not accept a file outside its project root."""
    inner = tmp_path / "project"
    inner.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Hello", encoding="utf-8")
    with pytest.raises(ValueError):
        Reader(inner).read_file(outside)
