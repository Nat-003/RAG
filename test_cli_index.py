"""Focused M3 tests for the tokenizer and BM25 retrieval."""
import math
import random
from collections import Counter

import pytest

from src.models import Chunk, MinimalSource
from src.retrieval.bm25 import BM25
from src.retrieval.tokenizer import tokenize


def chunk(cid: int, text: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source=MinimalSource(
        file_path=f"data/raw/test_{cid}.txt", first_character_index=0,
        last_character_index=len(text)))


def build(*chunks: Chunk) -> BM25:
    index = BM25(list(chunks))
    index.build()
    return index


@pytest.fixture
def toy() -> BM25:
    return build(chunk(0, "gpu memory memory"), chunk(1, "gpu worker"),
                 chunk(2, "worker"))


@pytest.mark.parametrize("text,expected", [
    ("GPU Memory", ["gpu", "memory"]),
    ("gpu_memory", ["gpu_memory", "gpu", "memory"]),
    ("gpuMemory", ["gpumemory", "gpu", "memory"]),
    ("GpuMemory", ["gpumemory", "gpu", "memory"]),
    ("HTTPServer", ["httpserver", "http", "server"]),
    ("float16", ["float16"]),
    ("Qwen3", ["qwen3"]),
    ("42 0 2026", ["42", "0", "2026"]),
    ("__init__", ["__init__", "init"]),
    ("memory memory", ["memory", "memory"]),
    ("gpu.memory,worker!", ["gpu", "memory", "worker"]),
    ("__ !!!", []), ("", []), ("café naïve", ["café", "naïve"]),
])
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_postings(toy: BM25) -> None:
    assert toy.inverted_index == {"gpu": {0: 1, 1: 1}, "memory": {0: 2},
                                  "worker": {1: 1, 2: 1}}


def test_document_lengths(toy: BM25) -> None:
    assert toy.document_lengths == {0: 3, 1: 2, 2: 1}


def test_document_count(toy: BM25) -> None:
    assert toy.total_documents == 3


def test_avgdl(toy: BM25) -> None:
    assert toy.avgdl == 2.0


def test_identifier_tokens_counted() -> None:
    index = build(chunk(0, "gpu_memory"))
    assert index.document_lengths == {0: 3}
    assert index.inverted_index == {"gpu_memory": {0: 1}, "gpu": {0: 1},
                                    "memory": {0: 1}}


def test_rebuild_clears_state(toy: BM25) -> None:
    previous = toy.search("gpu memory", 5)
    toy.build()
    assert toy.search("gpu memory", 5) == previous
    toy.chunks = [chunk(9, "fresh")]
    toy.build()
    assert toy.inverted_index == {"fresh": {9: 1}}
    assert toy.document_lengths == {9: 1}
    assert toy.total_documents == 1 and toy.avgdl == 1.0


def test_empty_corpus_rejected() -> None:
    with pytest.raises(ValueError):
        BM25([]).build()


def test_nonconsecutive_ids() -> None:
    index = build(chunk(21, "gpu"), chunk(900, "gpu"))
    assert index.document_lengths == {21: 1, 900: 1}
    assert [i for i, _ in index.search("gpu", 5)] == [21, 900]


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError):
        build(chunk(0, "gpu gpu"), chunk(0, "worker"))


def test_idf_rare(toy: BM25) -> None:
    assert toy.calculate_idf("memory") == pytest.approx(math.log(8 / 3))


def test_idf_common(toy: BM25) -> None:
    assert toy.calculate_idf("gpu") == pytest.approx(math.log(8 / 5))


def test_idf_rarity(toy: BM25) -> None:
    assert toy.calculate_idf("memory") > toy.calculate_idf("gpu") > 0


def test_idf_universal_term_positive() -> None:
    index = build(chunk(0, "gpu"), chunk(1, "gpu"))
    assert index.calculate_idf("gpu") == pytest.approx(math.log(6 / 5))


def test_score_known_numbers(toy: BM25) -> None:
    # N=3, DF(memory)=1, TF=2, dl=3, avgdl=2; denominator=4.0625.
    assert toy.calculate_score("memory", 0) == pytest.approx(math.log(8 / 3) * 5 / 4.0625)


def test_tf_saturation() -> None:
    index = build(chunk(0, "gpu filler filler filler"),
                  chunk(1, "gpu gpu filler filler"), chunk(2, "gpu gpu gpu filler"))
    a, b, c = (index.calculate_score("gpu", i) for i in range(3))
    assert 0 < c - b < b - a


def test_length_normalization() -> None:
    index = build(chunk(0, "gpu"), chunk(1, "gpu filler filler"))
    assert index.calculate_score("gpu", 0) > index.calculate_score("gpu", 1)


def test_or_search_and_accumulation(toy: BM25) -> None:
    result = toy.search("gpu memory worker", 10)
    assert {i for i, _ in result} == {0, 1, 2}
    scores = dict(result)
    assert scores[0] == pytest.approx(toy.calculate_score("gpu", 0) + toy.calculate_score("memory", 0))
    assert scores[1] == pytest.approx(toy.calculate_score("gpu", 1) + toy.calculate_score("worker", 1))
    assert scores[2] == pytest.approx(toy.calculate_score("worker", 2))


def test_ranking(toy: BM25) -> None:
    result = toy.search("gpu memory", 10)
    assert [i for i, _ in result] == [0, 1]
    assert result[0][1] > result[1][1] > 0


@pytest.mark.parametrize("k", [0, -1, -100])
def test_nonpositive_k(toy: BM25, k: int) -> None:
    assert toy.search("gpu", k) == []


@pytest.mark.parametrize("query", ["", "   ", "!!!", "unseen"])
def test_no_matches(toy: BM25, query: str) -> None:
    assert toy.search(query, 5) == []


def test_unknown_mixed(toy: BM25) -> None:
    assert toy.search("unknown gpu", 5) == toy.search("gpu", 5)


def test_duplicate_query_terms(toy: BM25) -> None:
    assert toy.search("gpu memory GPU gpu memory", 5) == toy.search("gpu memory", 5)


def test_top_k(toy: BM25) -> None:
    assert toy.search("gpu memory worker", 2) == toy.search("gpu memory worker", 10)[:2]


def test_large_k(toy: BM25) -> None:
    assert len(toy.search("memory", 10000)) == 1


def test_tie_breaker() -> None:
    index = build(chunk(8, "gpu"), chunk(2, "gpu"), chunk(5, "gpu"))
    assert [i for i, _ in index.search("gpu", 2)] == [2, 5]


def test_zero_token_corpus() -> None:
    index = build(chunk(0, "!!!"), chunk(1, "___"))
    assert index.avgdl == 0 and index.inverted_index == {}
    assert index.search("gpu", 5) == []


def test_mixed_zero_token_chunks() -> None:
    index = build(chunk(0, "!!!"), chunk(1, "gpu"))
    assert index.avgdl == .5
    assert [i for i, _ in index.search("gpu", 5)] == [1]


def test_unbuilt_index() -> None:
    assert BM25([chunk(0, "gpu")]).search("gpu", 5) == []


def test_repeatable_search(toy: BM25) -> None:
    first = toy.search("gpu memory worker", 3)
    for _ in range(10):
        assert toy.search("gpu memory worker", 3) == first


def test_search_does_not_mutate_index(toy: BM25) -> None:
    before = (dict(toy.document_lengths), {t: dict(p) for t, p in toy.inverted_index.items()}, toy.avgdl)
    toy.search("gpu worker", 5)
    assert (toy.document_lengths, toy.inverted_index, toy.avgdl) == before


def dense_reference(documents: dict[int, list[str]], terms: list[str], k: int) -> list[tuple[int, float]]:
    """Score all documents directly; no posting lists or BM25 class methods."""
    n = len(documents)
    avg = sum(map(len, documents.values())) / n
    counts = {i: Counter(words) for i, words in documents.items()}
    terms = list(dict.fromkeys(terms))
    results = []
    for cid, words in documents.items():
        score = 0.0
        for term in terms:
            tf = counts[cid][term]
            if tf:
                df = sum(term in doc for doc in documents.values())
                weight = math.log((n + 1) / (df + .5))
                score += weight * tf * 2.5 / (tf + 1.5 * (.25 + .75 * len(words) / avg))
        if score > 0:
            results.append((cid, score))
    return sorted(results, key=lambda row: (-row[1], row[0]))[:k]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_against_dense_reference(seed: int) -> None:
    rng = random.Random(seed)
    vocab = ["gpu", "memory", "worker", "cache", "tensor", "model", "attention"]
    documents = {i * 3 + 5: rng.choices(vocab, k=rng.randint(1, 25)) for i in range(50)}
    index = build(*(chunk(i, " ".join(words)) for i, words in documents.items()))
    for _ in range(25):
        terms = rng.choices(vocab + ["absent"], k=rng.randint(1, 5))
        k = rng.randint(1, 50)
        actual = index.search(" ".join(terms), k)
        expected = dense_reference(documents, terms, k)
        assert [i for i, _ in actual] == [i for i, _ in expected]
        assert [s for _, s in actual] == pytest.approx([s for _, s in expected])
