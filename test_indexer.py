"""Verify IndexStorage persistence, reload parity and lightweight validation."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.indexing.indexer import Indexer
from src.models import Chunk, MinimalSource
from src.retrieval.bm25 import BM25
from src.retrieval.tokenizer import tokenize
from src.storage.index_store import IndexStorage


def make_chunk(cid: int, text: str, start: int = 0) -> Chunk:
    """Create a chunk with a nonempty source span."""
    return Chunk(chunk_id=cid, text=text, source=MinimalSource(
        file_path=f"data/raw/chunk_{cid}.txt",
        first_character_index=start,
        last_character_index=start + len(text),
    ))


@pytest.fixture
def original() -> BM25:
    """Use nonconsecutive IDs, Unicode, mixed lengths and a zero-token chunk."""
    index = BM25([
        make_chunk(7, "gpu_memory memory café\r\n", 14),
        make_chunk(2, "gpu worker"),
        make_chunk(51, "worker HTTPServer Qwen3"),
        make_chunk(90, "!!!"),
    ])
    index.k1, index.b = 1.9, 0.4
    index.build()
    return index


@pytest.fixture
def storage(tmp_path: Path, original: BM25) -> IndexStorage:
    """Persist the fixture using the real storage API."""
    store = IndexStorage(tmp_path / "data" / "processed")
    store.save(original, 128)
    return store


def alter_json(path: Path, **changes: object) -> None:
    """Update selected fields without otherwise changing a valid JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_creates_storage_directory(tmp_path: Path) -> None:
    """Create missing parent directories."""
    path = tmp_path / "a" / "b"
    IndexStorage(path)
    assert path.is_dir()


def test_save_creates_three_artifacts(storage: IndexStorage) -> None:
    """Write the three agreed persistence files."""
    assert {p.name for p in storage.storage_path.iterdir()} == {
        "chunks.jsonl", "bm25_index.json", "manifest.json",
    }


def test_chunk_jsonl_round_trip(storage: IndexStorage, original: BM25) -> None:
    """Preserve IDs, Unicode, line endings, source paths and offsets."""
    lines = (storage.storage_path / "chunks.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line) for line in lines] == [
        c.model_dump() for c in original.chunks
    ]
    assert storage.load_chunks() == original.chunks


def test_bm25_state_round_trip(storage: IndexStorage, original: BM25) -> None:
    """Restore only reusable corpus statistics and integer dictionary keys."""
    path = storage.storage_path / "bm25_index.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {
        "inverted_index", "document_lengths", "total_documents", "avgdl",
    }
    assert set(raw["document_lengths"]) == {"7", "2", "51", "90"}
    state = storage.load_bm25()
    assert state.inverted_index == original.inverted_index
    assert state.document_lengths == original.document_lengths
    assert state.total_documents == original.total_documents
    assert state.avgdl == original.avgdl
    assert all(type(cid) is int for cid in state.document_lengths)
    assert all(type(cid) is int for p in state.inverted_index.values()
               for cid in p)


def test_manifest_round_trip(storage: IndexStorage) -> None:
    """Persist actual parameters and both compatibility versions."""
    assert storage.load_manifest().model_dump() == {
        "format_version": 1, "tokenizer_version": 1,
        "k1": 1.9, "b": 0.4, "max_chunk_size": 128,
    }


def test_restores_full_state(storage: IndexStorage, original: BM25) -> None:
    """Return a new BM25 instance with the exact saved state."""
    loaded = storage.load()
    assert isinstance(loaded, BM25) and loaded is not original
    for attr in ("chunks", "document_lengths", "inverted_index",
                 "total_documents", "avgdl", "k1", "b"):
        assert getattr(loaded, attr) == getattr(original, attr)


@pytest.mark.parametrize("query,k", [
    ("gpu memory", 5), ("HTTPServer", 1), ("worker", 10),
    ("gpu_memory café", 2), ("gpu GPU worker", 3),
    ("Qwen3", 5), ("unknown", 5), ("", 5), ("gpu", 0),
])
def test_search_parity(storage: IndexStorage, original: BM25,
                       query: str, k: int) -> None:
    """Require exact rank and score equality before and after loading."""
    assert storage.load().search(query, k) == original.search(query, k)


def test_load_does_not_reindex(storage: IndexStorage, original: BM25) -> None:
    """Fail if loading reads, chunks or tokenizes the corpus or builds BM25."""
    with patch("src.retrieval.bm25.BM25.build", side_effect=AssertionError), \
         patch("src.retrieval.bm25.tokenize", side_effect=AssertionError), \
         patch("src.indexing.reader.Reader.read_file", side_effect=AssertionError), \
         patch("src.indexing.reader.Reader.read_directory", side_effect=AssertionError), \
         patch("src.indexing.chunking.Chunker.chunk", side_effect=AssertionError):
        loaded = storage.load()
    assert loaded.search("gpu memory", 5) == original.search("gpu memory", 5)


def test_fresh_process_load(storage: IndexStorage, original: BM25) -> None:
    """Reload in an independent interpreter with no in-memory index."""
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from src.retrieval.bm25 import BM25\n"
        "from src.storage.index_store import IndexStorage\n"
        "from unittest.mock import patch\n"
        "with patch.object(BM25, 'build', side_effect=AssertionError):\n"
        "    idx = IndexStorage(Path(sys.argv[1])).load()\n"
        "print(json.dumps(idx.search('gpu memory worker', 5)))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script, str(storage.storage_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True,
        text=True, check=True, timeout=20,
    )
    assert json.loads(run.stdout) == [
        list(pair) for pair in original.search("gpu memory worker", 5)
    ]


def test_zero_token_index(tmp_path: Path) -> None:
    """Zero-token documents are valid and yield no matches."""
    index = BM25([make_chunk(4, "!!!"), make_chunk(9, "___")])
    index.build()
    store = IndexStorage(tmp_path)
    store.save(index, 2000)
    loaded = store.load()
    assert loaded.avgdl == 0.0
    assert loaded.document_lengths == {4: 0, 9: 0}
    assert loaded.inverted_index == {}
    assert loaded.search("gpu", 5) == []


@pytest.mark.parametrize("filename", [
    "chunks.jsonl", "bm25_index.json", "manifest.json",
])
def test_missing_file(storage: IndexStorage, filename: str) -> None:
    """Reject incomplete persistence bundles."""
    (storage.storage_path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        storage.load()


@pytest.mark.parametrize("filename", [
    "chunks.jsonl", "bm25_index.json", "manifest.json",
])
def test_malformed_json(storage: IndexStorage, filename: str) -> None:
    """Reject malformed JSON and JSONL."""
    (storage.storage_path / filename).write_text('{"unfinished":', encoding="utf-8")
    with pytest.raises(ValidationError):
        storage.load()


@pytest.mark.parametrize("field,value", [
    ("format_version", 2), ("tokenizer_version", 2),
    ("k1", 0), ("k1", -1), ("k1", float("inf")),
    ("b", -0.1), ("b", 1.1), ("max_chunk_size", 0),
    ("max_chunk_size", 2001),
])
def test_incompatible_manifest(storage: IndexStorage, field: str,
                               value: object) -> None:
    """Reject unsupported versions and invalid configuration."""
    alter_json(storage.storage_path / "manifest.json", **{field: value})
    with pytest.raises(ValidationError):
        storage.load()


@pytest.mark.parametrize("field,value", [
    ("total_documents", 0), ("avgdl", -1), ("avgdl", float("inf")),
    ("document_lengths", []), ("inverted_index", {"gpu": {"bad": 1}}),
])
def test_invalid_state(storage: IndexStorage, field: str, value: object) -> None:
    """Reject malformed persisted state using its existing Pydantic contract."""
    alter_json(storage.storage_path / "bm25_index.json", **{field: value})
    with pytest.raises(ValidationError):
        storage.load()


def test_invalid_chunk_source(storage: IndexStorage) -> None:
    """Reject invalid source offsets on deserialization."""
    path = storage.storage_path / "chunks.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["source"]["last_character_index"] = 0
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        storage.load()


def test_chunk_count_mismatch(storage: IndexStorage) -> None:
    """Trigger the first lightweight consistency check independently."""
    path = storage.storage_path / "chunks.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent number of chunks"):
        storage.load()


def test_document_length_count_mismatch(storage: IndexStorage) -> None:
    """Trigger the second lightweight consistency check independently."""
    path = storage.storage_path / "bm25_index.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["document_lengths"]["7"]
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent document lengths"):
        storage.load()


def test_indexer_to_storage(tmp_path: Path) -> None:
    """Check corpus-to-index integrity offline, then reload without source files."""
    corpus = tmp_path / "data" / "raw"
    corpus.mkdir(parents=True)
    (corpus / "gpu.py").write_bytes(b"def gpu_worker():\r\n    return 1\r\n")
    (corpus / "guide.md").write_text("# GPU memory\n\nCache worker\n", encoding="utf-8")
    (corpus / "notes.txt").write_text("café worker", encoding="utf-8")
    original = Indexer(tmp_path, corpus, 64).build()
    store = IndexStorage(tmp_path / "data" / "processed")
    store.save(original, 64)
    loaded = store.load()
    expected_postings: dict[str, dict[int, int]] = {}
    expected_lengths: dict[int, int] = {}
    for c in loaded.chunks:
        s = c.source
        text = (tmp_path / s.file_path).read_bytes().decode("utf-8")
        assert c.text == text[s.first_character_index:s.last_character_index]
        assert 0 < len(c.text) <= 64
        assert c.chunk_id not in expected_lengths
        tokens = tokenize(c.text)
        expected_lengths[c.chunk_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            expected_postings.setdefault(term, {})[c.chunk_id] = tf
    assert loaded.document_lengths == expected_lengths
    assert loaded.inverted_index == expected_postings
    assert loaded.total_documents == len(expected_lengths)
    assert loaded.avgdl == sum(expected_lengths.values()) / len(expected_lengths)
    for p in corpus.iterdir():
        p.unlink()
    corpus.rmdir()
    assert store.load().search("gpu worker", 5) == original.search("gpu worker", 5)
