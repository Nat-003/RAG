"""Integration checks for Reader -> Chunker -> BM25 orchestration."""
from pathlib import Path

import pytest

from src.indexing.indexer import Indexer
from src.retrieval.bm25 import BM25
from src.retrieval.tokenizer import tokenize


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / 'data' / 'raw'
    root.mkdir(parents=True)
    (root / 'code.py').write_bytes(
        "# café\r\ndef gpu_worker():\r\n    return 'gpu'\r\n\r\ndef other():\r\n    return 7\r\n".encode('utf-8')
    )
    (root / 'guide.md').write_text('# GPU memory\n\n' + 'memory cache ' * 20, encoding='utf-8')
    (root / 'notes.txt').write_text('uniquephrase worker', encoding='utf-8')
    (root / 'empty.md').write_bytes(b'')
    (root / 'bad.txt').write_bytes(b'\xff\xfe')
    (root / 'ignored.json').write_text('{}', encoding='utf-8')
    return root


def test_build_returns_searchable_bm25(tmp_path: Path, corpus: Path) -> None:
    bm25 = Indexer(tmp_path, corpus, 80).build()
    assert isinstance(bm25, BM25)
    assert bm25.total_documents == len(bm25.chunks) > 3
    result = bm25.search('uniquephrase', 5)
    assert len(result) == 1
    cid, score = result[0]
    assert bm25.chunks[cid].source.file_path == 'data/raw/notes.txt'
    assert score > 0


def test_global_ids_across_files(tmp_path: Path, corpus: Path) -> None:
    bm25 = Indexer(tmp_path, corpus, 40).build()
    assert [c.chunk_id for c in bm25.chunks] == list(range(len(bm25.chunks)))
    assert len(bm25.document_lengths) == bm25.total_documents
    assert {c.source.file_path for c in bm25.chunks} == {
        'data/raw/code.py', 'data/raw/guide.md', 'data/raw/notes.txt'
    }


def test_exact_offsets_and_chunk_size(tmp_path: Path, corpus: Path) -> None:
    bm25 = Indexer(tmp_path, corpus, 40).build()
    for chunk in bm25.chunks:
        original = (tmp_path / chunk.source.file_path).read_bytes().decode('utf-8')
        start = chunk.source.first_character_index
        end = chunk.source.last_character_index
        assert 0 <= start < end <= len(original)
        assert chunk.text == original[start:end]
        assert 0 < len(chunk.text) <= 40


def test_index_statistics_match_chunks(tmp_path: Path, corpus: Path) -> None:
    bm25 = Indexer(tmp_path, corpus, 80).build()
    expected_lengths = {c.chunk_id: len(tokenize(c.text)) for c in bm25.chunks}
    assert bm25.document_lengths == expected_lengths
    assert bm25.avgdl == pytest.approx(sum(expected_lengths.values()) / len(expected_lengths))
    assert all(bm25.chunks[i].chunk_id == i for p in bm25.inverted_index.values() for i in p)


def test_skipped_files_do_not_prevent_indexing(tmp_path: Path, corpus: Path) -> None:
    index = Indexer(tmp_path, corpus).build()
    assert not any(c.source.file_path.endswith(('empty.md', 'bad.txt', 'ignored.json')) for c in index.chunks)
    assert index.search('uniquephrase', 1)


def test_rebuild_is_repeatable(tmp_path: Path, corpus: Path) -> None:
    indexer = Indexer(tmp_path, corpus, 80)
    first = indexer.build()
    second = indexer.build()
    assert first.chunks == second.chunks
    assert first.inverted_index == second.inverted_index
    assert first.search('gpu memory', 5) == second.search('gpu memory', 5)
    assert first is not second


def test_empty_corpus(tmp_path: Path) -> None:
    root = tmp_path / 'empty'
    root.mkdir()
    with pytest.raises(ValueError, match='No chunks'):
        Indexer(tmp_path, root).build()


def test_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        Indexer(tmp_path, tmp_path / 'missing').build()


@pytest.mark.parametrize('size', [0, 2001])
def test_invalid_chunk_size(tmp_path: Path, corpus: Path, size: int) -> None:
    with pytest.raises(ValueError, match='max_size'):
        Indexer(tmp_path, corpus, size).build()
