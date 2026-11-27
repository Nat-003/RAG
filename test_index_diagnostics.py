"""Test the index command's orchestration and optional Fire entry point."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.cli import index
from src.indexing.indexer import Indexer
from src.storage.index_store import IndexStorage


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a miniature mixed-format corpus and run from its root."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "code.py").write_bytes(b"def gpu_worker():\r\n    return 42\r\n")
    (raw / "guide.md").write_text("# GPU\n\nMemory cache.", encoding="utf-8")
    (raw / "notes.txt").write_text("uniquephrase worker café", encoding="utf-8")
    (raw / "empty.txt").touch()
    return tmp_path


def test_default_index_roundtrip(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Build using default paths, then search after deleting the test corpus."""
    expected = Indexer(project, project / "data/raw").build()
    index()
    out = project / "data/processed"
    assert {p.name for p in out.iterdir()} == {"chunks.jsonl", "bm25_index.json", "manifest.json"}
    assert all(p.stat().st_size > 0 for p in out.iterdir())
    assert str(expected.total_documents) in capsys.readouterr().out
    shutil.rmtree(project / "data/raw")
    store = IndexStorage(out)
    restored = store.load()
    assert restored.chunks == expected.chunks
    assert restored.inverted_index == expected.inverted_index
    assert restored.document_lengths == expected.document_lengths
    assert restored.total_documents == expected.total_documents
    assert restored.avgdl == expected.avgdl
    assert store.load_manifest().max_chunk_size == 2000
    for query in ("uniquephrase", "gpu_worker", "memory", "worker", "absent"):
        assert restored.search(query, 5) == expected.search(query, 5)
    assert restored.search("uniquephrase", 5)


def test_configurable_paths(project: Path) -> None:
    """Respect custom corpus, storage and maximum chunk size."""
    raw = project / "custom_corpus"
    (project / "data/raw").rename(raw)
    index(corpus_path="custom_corpus", storage_path="custom_output", max_chunk_size=12)
    store = IndexStorage(project / "custom_output")
    assert store.load_manifest().max_chunk_size == 12
    loaded = store.load()
    assert loaded.chunks and all(0 < len(c.text) <= 12 for c in loaded.chunks)
    assert all(c.source.file_path.startswith("custom_corpus/") for c in loaded.chunks)
    assert not (project / "data/processed").exists()


def test_reindex_replaces_old_index(project: Path) -> None:
    """A completed second indexing run replaces the previous state."""
    index()
    store = IndexStorage(project / "data/processed")
    assert store.load().search("uniquephrase", 5)
    (project / "data/raw/notes.txt").write_text("newphrase", encoding="utf-8")
    index()
    restored = store.load()
    assert not restored.search("uniquephrase", 5)
    assert restored.search("newphrase", 5)


@pytest.mark.parametrize("size", [-1, 0, 2001])
def test_invalid_size(project: Path, size: int) -> None:
    """Reject unsupported integer limits before creating storage."""
    with pytest.raises(ValueError, match="Invalid chunk size"):
        index(max_chunk_size=size)
    assert not (project / "data/processed").exists()


@pytest.mark.parametrize("size", [1, 2000])
def test_valid_boundaries(project: Path, size: int) -> None:
    """Accept both supported boundaries and persist the configured limit."""
    index(max_chunk_size=size)
    store = IndexStorage(project / "data/processed")
    assert store.load_manifest().max_chunk_size == size
    assert all(0 < len(c.text) <= size for c in store.load().chunks)


def test_missing_corpus(project: Path) -> None:
    """Surface a missing directory as an OSError before saving."""
    with pytest.raises(NotADirectoryError):
        index(corpus_path="missing")
    assert not (project / "data/processed").exists()


def test_empty_corpus(project: Path) -> None:
    """Reject a corpus without indexable chunks."""
    (project / "empty").mkdir()
    with pytest.raises(ValueError, match="No chunks"):
        index(corpus_path="empty")
    assert not (project / "data/processed").exists()


def test_storage_is_file(project: Path) -> None:
    """Surface a storage-directory creation failure."""
    (project / "blocked").write_text("unchanged", encoding="utf-8")
    with pytest.raises(OSError):
        index(storage_path="blocked")
    assert (project / "blocked").read_text(encoding="utf-8") == "unchanged"


def run_fire(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real Python Fire CLI with the same interpreter as pytest."""
    if importlib.util.find_spec("fire") is None:
        pytest.skip("Python Fire is not installed in this environment")
    repo = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (repo, env.get("PYTHONPATH"))))
    env["PAGER"] = "cat"
    return subprocess.run([sys.executable, "-m", "src", *args], cwd=project,
                          env=env, text=True, capture_output=True, timeout=30)


def test_fire_help(project: Path) -> None:
    """Verify the six required commands are exposed by the real entry point."""
    result = run_fire(project, "--help")
    assert result.returncode == 0
    output = result.stdout + result.stderr
    for name in ("index", "search", "search_dataset", "answer", "answer_dataset", "evaluate"):
        assert name in output


def test_fire_valid_index(project: Path) -> None:
    """Verify the required index command creates a loadable index."""
    result = run_fire(project, "index", "--max_chunk_size", "2000")
    assert result.returncode == 0, result.stderr
    assert IndexStorage(project / "data/processed").load().search("uniquephrase", 1)


@pytest.mark.parametrize("args", [
    ("index", "--max_chunk_size", "0"),
    ("index", "--corpus_path", "missing"),
])
def test_fire_graceful_error(project: Path, args: tuple[str, ...]) -> None:
    """Verify the entry point exits nonzero without a traceback."""
    result = run_fire(project, *args)
    assert result.returncode == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stdout + result.stderr
