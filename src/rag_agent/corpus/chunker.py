"""
chunker.py
==========

Load Markdown documents and split them into DocumentChunk objects.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """Load documents and prepare chunks for embedding and storage."""

    # These values measure characters, not tokens.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """Load one file and create chunks with metadata and stable IDs."""
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be at least zero "
                "and smaller than chunk_size."
            )

        suffix = file_path.suffix.lower()

        if suffix == ".md":
            raw_chunks = self._chunk_markdown(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        elif suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "Use a .md file for this workshop."
            )

        metadata = self._infer_metadata(
            file_path,
            metadata_overrides,
        )

        chunks: list[DocumentChunk] = []

        for raw_chunk in raw_chunks:
            text = raw_chunk["text"].strip()

            if not text:
                continue

            chunk_id = VectorStoreManager.generate_chunk_id(
                metadata.source,
                text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=replace(
                        metadata,
                        related_topics=list(metadata.related_topics),
                    ),
                )
            )

        logger.info(
            "Created {} chunks from '{}'.",
            len(chunks),
            metadata.source,
        )

        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """Optional multi-file helper; not required for this workshop."""
        raise NotImplementedError(
            "Process each Markdown file using chunk_file()."
        )

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """PDF ingestion is outside the required workshop scope."""
        raise NotImplementedError(
            "PDF ingestion is not implemented. Upload a Markdown file."
        )

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Split Markdown by headings, then split oversized sections."""
        text = file_path.read_text(encoding="utf-8-sig")

        if not text.strip():
            return []

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        sections = header_splitter.split_text(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        raw_chunks: list[dict] = []

        for section in sections:
            header = " > ".join(
                section.metadata[key]
                for key in ("Header 1", "Header 2", "Header 3")
                if key in section.metadata
            )

            for chunk_text in text_splitter.split_text(
                section.page_content
            ):
                if chunk_text.strip():
                    raw_chunks.append(
                        {
                            "text": chunk_text,
                            "header": header,
                        }
                    )

        return raw_chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """Infer topic and difficulty from the filename, then apply overrides."""
        topic_names = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }

        difficulties = {
            "beginner",
            "intermediate",
            "advanced",
        }

        stem = file_path.stem.lower()
        parts = stem.split("_")

        difficulty = "beginner"

        if parts[-1] in difficulties:
            difficulty = parts.pop()

        topic_key = "".join(parts).replace("-", "").replace(" ", "")
        topic = topic_names.get(topic_key, "Unknown")

        values = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": False,
        }

        overrides = dict(overrides or {})

        unknown_fields = set(overrides) - set(values)

        if unknown_fields:
            raise ValueError(
                "Unknown metadata fields: "
                + ", ".join(sorted(unknown_fields))
            )

        values.update(overrides)

        # Normalize a recognized topic, including one supplied by an override.
        normalized_topic = (
            str(values["topic"])
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        values["topic"] = topic_names.get(
            normalized_topic,
            str(values["topic"]),
        )

        values["difficulty"] = str(
            values["difficulty"]
        ).lower()

        if values["difficulty"] not in difficulties:
            raise ValueError(
                "Difficulty must be beginner, intermediate, or advanced."
            )

        if "is_bonus" not in overrides:
            values["is_bonus"] = values["topic"] in {
                "SOM",
                "BoltzmannMachine",
                "GAN",
            }

        if values["topic"] == "Unknown":
            logger.warning(
                "Could not infer topic from '{}'. "
                "Use a topic-based filename or metadata override.",
                file_path.name,
            )

        return ChunkMetadata(**values)