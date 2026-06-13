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
    reference_sources: list[dict[str, str]] | None = None,
    query_terms: list[str] | None = None,
    query_text: str = "",
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """
    Retrieve known issues using the Chroma vector store.
    """
    store = get_vector_store()
    if reference_sources:
        parsed_sources = [ReferenceSource(**source) for source in reference_sources]
        store.index_reference_sources(parsed_sources)

    if not query_text and query_terms:
        query_text = " ".join(query_terms)

    if not query_text:
        return []

    return store.semantic_search(query_text=query_text, top_k=top_k)
