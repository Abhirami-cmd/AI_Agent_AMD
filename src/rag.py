from __future__ import annotations

from typing import Any

from src.models import ReferenceSource
from src.vector_store import ChromaVectorStore

_VECTOR_STORE: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = ChromaVectorStore()
    return _VECTOR_STORE


def retrieve_known_issues(
    reference_sources=None,
    query_terms=None,
    query_text="",
    top_k=4,
) -> list[dict[str, Any]]:
    store = get_vector_store()

    if reference_sources:
        parsed_sources = [ReferenceSource(**source) for source in reference_sources]
        store.index_reference_sources(parsed_sources)

    # 🔥 NEW: build multiple queries instead of one merged query
    queries = []

    if query_terms:
        # split structured queries instead of flattening everything
        for term in query_terms:
            queries.append(term)

    if query_text:
        queries.append(query_text)

    if not queries:
        return []

    # 🔥 retrieve per query
    all_results = []
    for q in queries:
        results = store.semantic_search(query_text=q, top_k=top_k)
        all_results.extend(results)

    # 🔥 deduplicate by title
    dedup = {}
    for r in all_results:
        key = r.get("title", "")
        if key not in dedup or r["score"] > dedup[key]["score"]:
            dedup[key] = r

    return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:top_k * 2]
