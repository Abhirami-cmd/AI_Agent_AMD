from __future__ import annotations

import os
import re
from hashlib import sha1
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import settings
from src.models import ReferenceSource


class ChromaVectorStore:
    def __init__(self, persist_dir: str | None = None) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = self._build_client()
        self._collection = self._client.get_or_create_collection(
            name="known_issues",
            metadata={"hnsw:space": "cosine"},
        )

    def _build_client(self) -> chromadb.Client:
        try:
            chroma_settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
            )
            return chromadb.Client(chroma_settings)
        except Exception:
            return chromadb.Client()

    def _extract_issue_blocks(self, text: str, source: str) -> list[tuple[str, str, str]]:
        blocks: list[tuple[str, str, str]] = []
        for raw_line in re.split(r"\n|(?<=\.)\s+(?=[A-Z][A-Za-z ]+:)", text):
            line = raw_line.strip(" -")
            if ":" not in line:
                continue
            title, body = line.split(":", 1)
            if len(title.split()) > 12:
                continue
            blocks.append((title.strip(), body.strip(), source))
        return blocks

    def index_reference_sources(self, reference_sources: list[ReferenceSource]) -> None:
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        ids: list[str] = []

        for source_index, source in enumerate(reference_sources):
            blocks = self._extract_issue_blocks(source.text, source.name)
            if not blocks:
                blocks = [(source.name, source.text, source.name)]

            for block_index, (title, body, block_source) in enumerate(blocks):
                digest = sha1(f"{source.name}:{title}:{body}".encode("utf-8")).hexdigest()[:16]
                doc_id = f"{source.name}_{source_index}_{block_index}_{digest}"
                documents.append(f"{title}. {body}")
                metadatas.append({
                    "title": title,
                    "body": body,
                    "source": block_source,
                    "path": source.path,
                })
                ids.append(doc_id)

        if not documents:
            return

        existing_ids: set[str] = set()
        try:
            current = self._collection.get()
            if isinstance(current, dict) and "ids" in current:
                existing_ids.update(str(item) for item in current["ids"])
        except Exception:
            existing_ids = set()

        filtered_documents: list[str] = []
        filtered_metadatas: list[dict[str, str]] = []
        filtered_ids: list[str] = []
        for doc, metadata, doc_id in zip(documents, metadatas, ids):
            if doc_id in existing_ids:
                continue
            filtered_documents.append(doc)
            filtered_metadatas.append(metadata)
            filtered_ids.append(doc_id)

        if filtered_documents:
            self._collection.add(
                documents=filtered_documents,
                metadatas=filtered_metadatas,
                ids=filtered_ids,
            )

    def semantic_search(self, query_text: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not query_text.strip():
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )

        if results is None:
            return []

        if hasattr(results, "to_dict") and not isinstance(results, dict):
            try:
                results = results.to_dict(orient="list")
            except Exception:
                try:
                    results = dict(results)
                except Exception:
                    return []

        issues: list[dict[str, Any]] = []
        ids = results.get("ids") if isinstance(results, dict) else None
        metadatas = results.get("metadatas") if isinstance(results, dict) else None
        distances = results.get("distances") if isinstance(results, dict) else None

        if not ids or not ids[0]:
            return []

        for idx, doc_id in enumerate(ids[0]):
            metadata = metadatas[0][idx] if metadatas else {}
            distance = distances[0][idx] if distances else 0
            similarity = max(0.0, 1.0 - float(distance))
            issues.append(
                {
                    "id": doc_id,
                    "title": metadata.get("title", ""),
                    "body": metadata.get("body", ""),
                    "score": similarity,
                    "source": metadata.get("source", "Chroma Vector DB"),
                    "path": metadata.get("path", ""),
                }
            )
        return issues
