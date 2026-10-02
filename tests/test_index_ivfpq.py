from __future__ import annotations

import numpy as np

from aic2026.retrieval.index import VectorIndex


class _ApproxSearch:
    def search(self, queries, k):
        # Deliberately return approximate order that is worse than exact cosine.
        ids = np.tile(np.array([1, 2, 0], dtype=np.int64)[:k], (len(queries), 1))
        scores = np.tile(np.array([0.95, 0.85, 0.10], dtype=np.float32)[:k], (len(queries), 1))
        return scores, ids


def test_ivfpq_candidates_are_rescored_against_original_vectors():
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32
    )
    index = VectorIndex(vectors, assume_normalized=True)
    index._faiss = _ApproxSearch()
    index._faiss_approx = True

    ids, scores = index.search(np.array([1.0, 0.0], dtype=np.float32), k=1)

    assert ids.tolist() == [0]
    assert scores.tolist() == [1.0]


def test_filtered_exact_search_ranks_only_selected_rows():
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32
    )
    index = VectorIndex(vectors, assume_normalized=True)

    ids, scores = index.search_filtered_indices(
        np.array([1.0, 0.0], dtype=np.float32), 2, np.array([1, 2])
    )

    assert ids.tolist() == [2, 1]
    assert np.allclose(scores, [0.8, 0.0])
