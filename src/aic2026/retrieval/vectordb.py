from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.models import FrameRecord


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row in place semantics, matching VectorIndex."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class ChromaVectorStore:
    """Persistent in-process ChromaDB backend, drop-in for ``VectorIndex.search``.

    ``search`` returns ``(ids, scores)`` where ``ids`` are **manifest indices**
    (identical semantics to ``VectorIndex.search``) and ``scores`` are cosine
    similarities. Chroma's internal IDs are ``str(record.vector_id)``; because
    ``vector_id`` is assigned in manifest order, ``int(chroma_id) == manifest index``.

    ChromaDB is optional. Importing this module is always safe; construction
    (``__init__`` / ``from_*``) raises ``ImportError`` when ``chromadb`` is missing.
    """

    _DEFAULT_METADATA_FIELDS = (
        "video_id",
        "frame_id",
        "keyframe_path",
        "object_labels",
        "metadata_keywords",
        "title",
        "description",
    )

    def __init__(
        self,
        vectors: np.ndarray,
        manifest: list[FrameRecord],
        collection_name: str = "aic2026_frames",
        persist_dir: str | Path = "data/indexes/chroma",
        metadata_fields: tuple[str, ...] | None = None,
        space: str = "cosine",
        *,
        _open_existing: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for ChromaVectorStore: "
                "uv sync --extra retrieval"
            ) from exc

        self.manifest = manifest
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.space = space
        self.metadata_fields = metadata_fields or self._DEFAULT_METADATA_FIELDS
        self._ordered_ids = [str(record.vector_id) for record in manifest]
        self._id_to_manifest_index: dict[str, int] | None = None
        self._manifest_index_to_id: dict[int, str] | None = None
        self._vectors_cache: np.ndarray | None = None

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
        )

        if _open_existing:
            # Query-time open: never add, just validate the collection is usable.
            if self._collection.count() == 0:
                raise ValueError(
                    f"Chroma collection '{collection_name}' is empty at "
                    f"{self.persist_dir}. Build it first (aic2026 build-chroma-index)."
                )
            return

        if len(vectors) != len(manifest):
            raise ValueError("Feature count must equal manifest record count")

        if self._collection.count() == 0 and len(vectors) > 0:
            self.add(
                ids=self._ordered_ids,
                embeddings=_normalize_rows(vectors),
                metadatas=[self._metadata_for(record) for record in manifest],
            )
        elif self._collection.count() != len(vectors):
            raise ValueError(
                "Chroma collection already exists with a different size "
                f"({self._collection.count()} vs {len(vectors)}). "
                "Delete the collection or point to a fresh persist_dir."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_npy(
        cls,
        features: Path,
        manifest: list[FrameRecord],
        persist_dir: str | Path = "data/indexes/chroma",
        collection_name: str = "aic2026_frames",
        space: str = "cosine",
        **kwargs: Any,
    ) -> ChromaVectorStore:
        """Build a fresh collection from a feature ``.npy`` file + manifest."""
        vectors = np.load(features)
        return cls(
            vectors=vectors,
            manifest=manifest,
            persist_dir=persist_dir,
            collection_name=collection_name,
            space=space,
            **kwargs,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: list[FrameRecord],
        persist_dir: str | Path = "data/indexes/chroma",
        collection_name: str = "aic2026_frames",
        space: str = "cosine",
    ) -> ChromaVectorStore:
        """Open an existing collection (query time); raises if collection missing."""
        store = cls(
            vectors=np.empty((0, 0), dtype=np.float32),
            manifest=manifest,
            persist_dir=persist_dir,
            collection_name=collection_name,
            space=space,
            _open_existing=True,
        )
        return store

    # ------------------------------------------------------------------
    # Compatibility surface
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> np.ndarray:
        """Materialized, manifest-ordered embeddings.

        Chroma ``get(include=["embeddings"])`` is not guaranteed to be ordered,
        so rows are reordered to match ``manifest`` order. This keeps
        ``RetrievalPipeline.retrieve_trake``'s fancy-index (``vectors[indices]``)
        correct with both backends.
        """
        if self._vectors_cache is None:
            result = self._collection.get(include=["embeddings"])
            raw = {
                chroma_id: emb
                for chroma_id, emb in zip(
                    result["ids"],
                    result["embeddings"],
                )
            }
            if len(raw) != len(self._ordered_ids):
                raise ValueError(
                    "Chroma collection size does not match the manifest. "
                    "Rebuild the index (aic2026 build-chroma-index)."
                )
            rows = [np.asarray(raw[chroma_id], dtype=np.float32) for chroma_id in self._ordered_ids]
            if not rows:
                self._vectors_cache = np.empty((0, 0), dtype=np.float32)
            else:
                self._vectors_cache = np.stack(rows)
        return self._vectors_cache

    @property
    def id_to_manifest_index(self) -> dict[str, int]:
        if self._id_to_manifest_index is None:
            self._id_to_manifest_index = {
                chroma_id: index
                for index, chroma_id in enumerate(self._ordered_ids)
            }
        return self._id_to_manifest_index

    @property
    def manifest_index_to_id(self) -> dict[int, str]:
        if self._manifest_index_to_id is None:
            self._manifest_index_to_id = {
                index: chroma_id
                for index, chroma_id in enumerate(self._ordered_ids)
            }
        return self._manifest_index_to_id

    def __len__(self) -> int:
        return len(self._ordered_ids)

    @property
    def collection_count(self) -> int:
        """Number of vectors physically stored in the Chroma collection."""
        return int(self._collection.count())

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert/upsert vectors with their metadata.

        Chroma caps a single ``upsert`` at 5461 rows, so large manifests (e.g.
        177k frames) are split into safe batches automatically.
        """
        # Chroma hard limit is 5461 rows/upsert; keep a safety margin.
        _CHROMA_MAX_BATCH = 5000
        normalized = _normalize_rows(embeddings)
        emb_rows = normalized.tolist()
        total = len(ids)
        if total <= _CHROMA_MAX_BATCH:
            self._collection.upsert(ids=list(ids), embeddings=emb_rows, metadatas=list(metadatas))
            return
        for start in range(0, total, _CHROMA_MAX_BATCH):
            end = start + _CHROMA_MAX_BATCH
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=emb_rows[start:end],
                metadatas=metadatas[start:end],
            )

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(manifest_indices, cosine_similarities)`` for the top-k.

        Chroma returns cosine *distance* for ``hnsw:space="cosine"``; we convert
        back to similarity (``1 - distance``) so scores descend like
        ``VectorIndex.search``.
        """
        if k <= 0 or self._collection.count() == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        result = self._collection.query(
            query_embeddings=[q.tolist()],
            n_results=min(int(k), self._collection.count()),
        )
        chroma_ids = result["ids"][0]
        distances = result["distances"][0]

        mapping = self.id_to_manifest_index
        ids = np.asarray([mapping[chroma_id] for chroma_id in chroma_ids], dtype=int)
        scores = np.asarray(
            [1.0 - float(distance) for distance in distances],
            dtype=np.float32,
        )
        return ids, scores

    def search_filtered(
        self,
        query: np.ndarray,
        k: int,
        video_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vector search restricted to *video_ids* via Chroma ``where`` filter.

        Unlike the FAISS/NumPy backend, Chroma pushes the video constraint down
        into the ANN scan, so only matching frames are ever visited. Pass
        ``None`` for no filtering.
        """
        if k <= 0 or self._collection.count() == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        where = {"video_id": {"$in": sorted(video_ids)}} if video_ids else None
        result = self._collection.query(
            query_embeddings=[q.tolist()],
            n_results=min(int(k), self._collection.count()),
            where=where,
        )
        chroma_ids = result["ids"][0]
        distances = result["distances"][0]

        mapping = self.id_to_manifest_index
        ids = np.asarray([mapping[chroma_id] for chroma_id in chroma_ids], dtype=int)
        scores = np.asarray(
            [1.0 - float(distance) for distance in distances],
            dtype=np.float32,
        )
        return ids, scores

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_for(record: FrameRecord) -> dict[str, Any]:
        """JSON-safe metadata dict (Chroma rejects list/None values in places)."""
        metadata: dict[str, Any] = {
            "video_id": record.video_id,
            "frame_id": int(record.frame_id),
            "keyframe_path": record.keyframe_path or "",
        }
        if record.object_labels:
            metadata["object_labels"] = json.dumps(
                list(record.object_labels),
                ensure_ascii=False,
            )
        if record.metadata_keywords:
            metadata["metadata_keywords"] = json.dumps(
                list(record.metadata_keywords),
                ensure_ascii=False,
            )
        if record.title:
            metadata["title"] = record.title
        if record.description:
            metadata["description"] = record.description
        return metadata

    @staticmethod
    def available() -> bool:
        try:
            import chromadb  # noqa: F401
            return True
        except ImportError:
            return False
