"""Regressions for the M3 progress bars and failed-read diagnostics."""
from pathlib import Path

import pytest

from src.cli import index
from src.indexing.indexer import Indexer
from src.storage.index_store import IndexStorage


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Create valid, empty and invalid UTF-8 inputs."""
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "valid.txt").write_text("gpu memory cache", encoding="utf-8")
    (raw / "empty.md").touch()
    (raw / "invalid.txt").write_bytes(b"\xff\xfe")
    return raw


def test_failed_read_records_paths_and_reasons(corpus: Path, tmp_path: Path) -> None:
    """Expose both unsuccessful reads without losing the valid file."""
    indexer = Indexer(tmp_path, corpus)
    bm25 = indexer.build()
    assert bm25.total_documents == 1
    assert set(indexer.failed_read) == {corpus / "empty.md", corpus / "invalid.txt"}
    assert "Empty file" in indexer.failed_read[corpus / "empty.md"]
    assert "utf-8" in indexer.failed_read[corpus / "invalid.txt"].lower()


def test_failed_read_clears_after_rebuild(corpus: Path, tmp_path: Path) -> None:
    """A repaired corpus no longer reports previous read issues."""
    indexer = Indexer(tmp_path, corpus)
    indexer.build()
    (corpus / "empty.md").write_text("new document", encoding="utf-8")
    (corpus / "invalid.txt").write_text("repaired", encoding="utf-8")
    assert indexer.build().total_documents == 3
    assert not indexer.failed_read


def test_failed_read_is_instance_local(corpus: Path, tmp_path: Path) -> None:
    """Separate Indexer instances own their own dictionaries."""
    first = Indexer(tmp_path, corpus)
    second = Indexer(tmp_path, corpus)
    first.build()
    assert len(first.failed_read) == 2
    assert not second.failed_read
    assert first.failed_read is not second.failed_read


def test_cli_diagnostics_and_progress(corpus: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    """Report read issues and all three progress stages; save a usable index."""
    monkeypatch.chdir(tmp_path)
    index()
    captured = capsys.readouterr()
    assert "Read issues: 2" in captured.err
    assert "empty.md" in captured.err and "invalid.txt" in captured.err
    for label in ("Reading and chunking", "Building BM25 index", "Saving chunks"):
        assert label in captured.err
    assert "Read issues" not in captured.out
    assert "Saving BM25 statistics..." in captured.out
    assert "Indexed 1 chunks" in captured.out
    assert "Index saved to" in captured.out
    assert IndexStorage(tmp_path / "data/processed").load().search("gpu", 1)


def test_cli_no_warning_for_clean_corpus(corpus: Path, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    """Avoid reporting read issues when all files are readable."""
    (corpus / "empty.md").unlink()
    (corpus / "invalid.txt").unlink()
    monkeypatch.chdir(tmp_path)
    index()
    assert "Read issues:" not in capsys.readouterr().err
