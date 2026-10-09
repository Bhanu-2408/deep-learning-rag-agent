"""
store.py
========

Manage persistent ChromaDB storage, ingestion, and retrieval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """Manage the persistent ChromaDB collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """Create or open the persistent database and collection."""
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(db_path)
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "Opened collection '{}' with {} chunks.",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not initialise ChromaDB at "
                f"{self._settings.chroma_db_path}: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """Generate a stable ID from the source and chunk text."""
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """Return True if the chunk ID already exists."""
        result = self._collection.get(
            ids=[chunk_id],
            include=[],
        )
        return chunk_id in result["ids"]

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """Embed and store new chunks in batches of up to 100."""
        result = IngestionResult()
        pending: list[DocumentChunk] = []
        seen_ids: set[str] = set()

        for chunk in chunks:
            try:
                if not chunk.chunk_text.strip():
                    raise ValueError("Chunk text is empty.")

                if (
                    chunk.chunk_id in seen_ids
                    or self.check_duplicate(chunk.chunk_id)
                ):
                    result.skipped += 1
                    continue

                seen_ids.add(chunk.chunk_id)
                pending.append(chunk)

            except Exception as exc:
                result.errors.append(
                    f"Chunk {chunk.chunk_id}: {exc}"
                )

        batch_size = 100

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            try:
                texts = [chunk.chunk_text for chunk in batch]
                embeddings = self._embeddings.embed_documents(texts)

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[
                        chunk.metadata.to_dict()
                        for chunk in batch
                    ],
                )

                result.ingested += len(batch)

                for chunk in batch:
                    source = chunk.metadata.source

                    if source not in result.document_ids:
                        result.document_ids.append(source)

            except Exception as exc:
                for chunk in batch:
                    result.errors.append(
                        f"Chunk {chunk.chunk_id}: {exc}"
                    )

                logger.error("Ingestion batch failed: {}", exc)

        logger.info(
            "Ingestion finished: {} added, {} skipped, {} errors.",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks, applying metadata and score filters."""
        if not query_text.strip():
            return []

        total_chunks = self._collection.count()

        if total_chunks == 0:
            return []

        requested_k = (
            self._settings.retrieval_k
            if k is None
            else k
        )

        if requested_k < 1:
            raise ValueError("k must be at least 1.")

        filters = []

        if topic_filter:
            filters.append({"topic": {"$eq": topic_filter}})

        if difficulty_filter:
            filters.append(
                {"difficulty": {"$eq": difficulty_filter}}
            )

        if len(filters) == 2:
            where_filter = {"$and": filters}
        elif len(filters) == 1:
            where_filter = filters[0]
        else:
            where_filter = None

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(requested_k, total_chunks),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            # Cosine distance = 1 - cosine similarity.
            score = 1.0 - float(distance)

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        retrieved.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return retrieved

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """List unique source documents and their chunk counts."""
        results = self._collection.get(
            include=["metadatas"]
        )

        documents: dict[str, dict] = {}

        for metadata in results["metadatas"]:
            source = metadata["source"]

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata["topic"],
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda document: document["source"],
        )

    def get_document_chunks(
        self,
        source: str,
    ) -> list[DocumentChunk]:
        """Retrieve all stored chunks for one source document."""
        results = self._collection.get(
            where={"source": {"$eq": source}},
            include=["documents", "metadatas"],
        )

        rows = list(
            zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        )

        # Use chunk_index if ingestion metadata includes it.
        # Otherwise retain the database's returned order.
        rows.sort(
            key=lambda row: int(
                row[2].get("chunk_index", 0)
            )
        )

        return [
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=text,
                metadata=ChunkMetadata.from_dict(metadata),
            )
            for chunk_id, text, metadata in rows
        ]

    def get_collection_stats(self) -> dict:
        """Return chunk count, topics, sources, and bonus-topic status."""
        results = self._collection.get(
            include=["metadatas"]
        )

        metadatas = results["metadatas"]

        topics = sorted({
            metadata["topic"]
            for metadata in metadatas
        })

        sources = sorted({
            metadata["source"]
            for metadata in metadatas
        })

        bonus_topics_present = any(
            metadata["topic"] in {
                "SOM",
                "BoltzmannMachine",
                "GAN",
            }
            or str(
                metadata.get("is_bonus", "false")
            ).lower() == "true"
            for metadata in metadatas
        )

        return {
            "total_chunks": self._collection.count(),
            "topics": topics,
            "sources": sources,
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(self, source: str) -> int:
        """Delete all chunks for a source and return the count."""
        where_filter = {"source": {"$eq": source}}

        results = self._collection.get(
            where=where_filter,
            include=[],
        )

        count = len(results["ids"])

        if count:
            self._collection.delete(
                where=where_filter
            )

        logger.info(
            "Deleted {} chunks from '{}'.",
            count,
            source,
        )

        return count