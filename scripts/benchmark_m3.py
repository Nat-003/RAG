"""Measure M3 build, persistence and BM25 retrieval performance."""
import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

from src.indexing.indexer import Indexer
from src.storage.index_store import IndexStorage


def sample_queries() -> list[str]:
    """Generate 200 distinct, non-reference questions."""
    topics = (
        "gpu memory", "kv cache", "attention", "tokenizer", "scheduler",
        "tensor parallel", "pipeline parallel", "model loading", "quantization",
        "worker", "distributed inference", "sampling", "batch size",
        "request processing", "prefill", "decode", "block allocation",
        "prefix caching", "LoRA", "OpenAI server",
    )
    templates = (
        "How does vLLM handle {}?", "Where is {} implemented?",
        "What configuration controls {}?", "How can I enable {}?",
        "What is the role of {} in inference?", "Which class manages {}?",
        "How does the engine initialize {}?", "What limits affect {}?",
        "How are errors handled for {}?", "Explain the default behavior of {}.",
    )
    return [template.format(topic) for topic in topics for template in templates]


def load_questions(paths: list[Path]) -> list[str]:
    """Read 200 nonempty questions from the given reference datasets."""
    questions: list[str] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        questions.extend(record["question"] for record in data["rag_questions"])
    if len(questions) != 200 or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Expected exactly 200 nonempty reference questions")
    return questions


def peak_rss_mib() -> float | None:
    """Read peak process resident memory where the OS exposes it."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            """Windows process memory counters."""
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if get_memory(current(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 ** 2)
        return None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


def main() -> None:
    """Load the index once and time 200 searches; optionally rebuild first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", type=Path, default=Path("data/raw"))
    parser.add_argument("--storage-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-chunk-size", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--sample-queries", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Replace saved index")
    args = parser.parse_args()
    if not 1 <= args.max_chunk_size <= 2000 or args.k <= 0:
        parser.error("Invalid chunk size or k")
    if args.sample_queries and args.dataset:
        parser.error("Choose sample queries or reference datasets, not both")
    paths = args.dataset or [
        Path("data/datasets/UnansweredQuestions/dataset_docs_public.json"),
        Path("data/datasets/UnansweredQuestions/dataset_code_public.json"),
    ]
    try:
        questions = sample_queries() if args.sample_queries else load_questions(paths)
        print("Questions:", "200 sample (NOT reference) questions" if args.sample_queries else f"200 reference questions from {paths}", flush=True)
        storage = IndexStorage(args.storage_path)
        if args.rebuild:
            indexer = Indexer(Path.cwd(), args.corpus_path, args.max_chunk_size)
            start = perf_counter()
            built = indexer.build()
            built_at = perf_counter()
            storage.save(built, args.max_chunk_size)
            saved_at = perf_counter()
            print(f"Reader + Chunker + BM25: {built_at - start:.3f} s")
            print(f"Save: {saved_at - built_at:.3f} s")
            print(f"Build + save: {saved_at - start:.3f} s")
            print(f"Read issues: {len(indexer.failed_read)}")
            del built, indexer
            gc.collect()
        files = [args.storage_path / name for name in ("chunks.jsonl", "bm25_index.json", "manifest.json")]
        sizes = [(p.name, p.stat().st_size) for p in files]
        start = perf_counter()
        bm25 = storage.load()
        load_s = perf_counter() - start
        print(f"Chunks loaded: {bm25.total_documents:,}")
        print(f"Vocabulary: {len(bm25.inverted_index):,}")
        print(f"Postings: {sum(len(p) for p in bm25.inverted_index.values()):,}")
        for name, size in sizes:
            print(f"{name}: {size:,} bytes ({size / 1024**2:.3f} MiB)")
        print(f"Total index size: {sum(size for _, size in sizes) / 1024**2:.3f} MiB")
        print(f"Load: {load_s:.3f} s", flush=True)
        durations: list[float] = []
        results: list[list[tuple[int, float]]] = []
        start = perf_counter()
        for question in questions:
            query_start = perf_counter()
            results.append(bm25.search(question, args.k))
            durations.append(perf_counter() - query_start)
        search_s = perf_counter() - start
        print(f"200 queries (k={args.k}): {search_s:.3f} s")
        print(f"Load + queries: {load_s + search_s:.3f} s")
        print(f"Mean latency: {statistics.mean(durations) * 1000:.3f} ms")
        print(f"P95 latency: {sorted(durations)[math.ceil(len(durations) * .95) - 1] * 1000:.3f} ms")
        print(f"Nonempty results: {sum(bool(r) for r in results)}/200")
        peak = peak_rss_mib()
        if peak is not None:
            print(f"Peak process RSS: {peak:.1f} MiB (includes build if --rebuild)")
        print("BM25-only benchmark; source resolution and result JSON are not yet measured.")
        print("No Recall@k score is computed.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\nIf reference datasets are unavailable, pass --sample-queries.\n")


if __name__ == "__main__":
    main()
