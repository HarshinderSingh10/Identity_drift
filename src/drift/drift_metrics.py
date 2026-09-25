"""Mathematical metrics for face-embedding drift analysis.

This module contains deterministic NumPy-based utilities for comparing two or
more face embeddings. It does not define identity-stability thresholds or
re-enrollment decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_embedding(embedding: Any) -> np.ndarray:
    """Validate and return an embedding as a 1-D ``float32`` NumPy array.

    Face embeddings represent an identity as a numeric vector. This helper
    accepts NumPy arrays or array-like input, converts values to ``float32``,
    flattens the result to one dimension, and rejects values that cannot be used
    safely in distance calculations.

    Raises
    ------
    ValueError
        If the embedding is empty, non-numeric, not finite, or has an
        incompatible scalar shape.
    """

    try:
        array = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding must be numeric and convertible to float32.") from exc

    if array.ndim == 0:
        raise ValueError("Embedding must be a 1-D vector or array-like sequence.")

    flattened = array.reshape(-1).astype(np.float32, copy=False)

    if flattened.size == 0:
        raise ValueError("Embedding must not be empty.")
    if not np.all(np.isfinite(flattened)):
        raise ValueError("Embedding must not contain NaN or infinite values.")

    return flattened


def embedding_norm(embedding: Any) -> float:
    """Return the L2 norm of an embedding.

    The L2 norm measures vector magnitude. ArcFace embeddings are commonly
    normalized near 1.0 before cosine comparison, but this function also handles
    non-normalized vectors.
    """

    vector = validate_embedding(embedding).astype(np.float64, copy=False)
    return float(np.linalg.norm(vector))


def is_normalized(embedding: Any, tolerance: float = 1e-5) -> bool:
    """Return whether an embedding has L2 norm close to 1.0.

    In this project, normalized embeddings make cosine-based identity drift
    comparisons easier to interpret. ``True`` means the vector magnitude is
    within ``tolerance`` of 1.0.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    return bool(abs(embedding_norm(embedding) - 1.0) <= tolerance)


def cosine_similarity(embedding_a: Any, embedding_b: Any) -> float:
    """Return cosine similarity between two embeddings.

    Mathematical definition:
    ``cos(A, B) = (A dot B) / (||A|| ||B||)``.

    In identity drift analysis, values near 1 mean the embeddings point in a
    very similar direction, values near 0 mean little directional similarity,
    and values near -1 mean opposite directions. The function handles
    non-normalized embeddings and clips the result to ``[-1, 1]`` to avoid tiny
    floating-point overshoots.
    """

    vector_a, vector_b = _validate_pair(embedding_a, embedding_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_distance(embedding_a: Any, embedding_b: Any) -> float:
    """Return cosine distance between two embeddings.

    Mathematical definition: ``distance = 1 - cosine_similarity``.

    For identity drift work, smaller values indicate less directional change
    between embeddings, while larger values indicate greater embedding change.
    """

    return float(1.0 - cosine_similarity(embedding_a, embedding_b))


def identity_drift(embedding_reference: Any, embedding_current: Any) -> float:
    """Return the primary identity drift score between two embeddings.

    Mathematical definition:
    ``drift = 1 - cosine_similarity(reference, current)``.

    A drift value near 0 means the current embedding is very close to the
    reference embedding in cosine space. This function does not assign semantic
    thresholds such as acceptable or unacceptable drift.
    """

    return cosine_distance(embedding_reference, embedding_current)


def euclidean_distance(embedding_a: Any, embedding_b: Any) -> float:
    """Return L2/Euclidean distance between two embeddings.

    Mathematical definition: ``||A - B||``.

    In identity drift analysis, this measures direct geometric separation in
    embedding space. For L2-normalized embeddings, it complements cosine drift
    while still remaining threshold-free at this stage.
    """

    vector_a, vector_b = _validate_pair(embedding_a, embedding_b)
    return float(np.linalg.norm(vector_a - vector_b))


def batch_cosine_similarity(reference_embedding: Any, embeddings: Any) -> np.ndarray:
    """Return cosine similarities from one reference to a batch of embeddings.

    The reference must be one embedding vector and ``embeddings`` must be a 2-D
    array with one embedding per row. Results preserve the input row order.
    Non-normalized embeddings are handled by normalizing during the calculation.
    """

    reference, batch = _validate_reference_and_batch(reference_embedding, embeddings)

    reference_norm = np.linalg.norm(reference)
    batch_norms = np.linalg.norm(batch, axis=1)
    if reference_norm == 0.0 or np.any(batch_norms == 0.0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    similarities = batch @ reference / (batch_norms * reference_norm)
    return np.clip(similarities, -1.0, 1.0).astype(np.float64, copy=False)


def batch_identity_drift(reference_embedding: Any, embeddings: Any) -> np.ndarray:
    """Return identity drift values from one reference to a batch of embeddings.

    Mathematical definition for each row:
    ``drift = 1 - cosine_similarity(reference, row)``.

    The returned 1-D array contains one drift value per input embedding and does
    not apply project thresholds or labels.
    """

    return 1.0 - batch_cosine_similarity(reference_embedding, embeddings)


def summarize_drift(drift_values: Any) -> dict[str, float | int]:
    """Return descriptive statistics for observed drift values.

    The summary describes measured drift values only. It intentionally avoids
    labels or thresholds, because re-enrollment boundaries should be determined
    later from experimental data.
    """

    values = validate_embedding(drift_values).astype(np.float64, copy=False)

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "standard_deviation": float(np.std(values)),
        "percentile_25": float(np.percentile(values, 25)),
        "percentile_75": float(np.percentile(values, 75)),
        "percentile_95": float(np.percentile(values, 95)),
    }


def _validate_pair(embedding_a: Any, embedding_b: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate two embeddings and return float64 vectors with matching shapes."""

    vector_a = validate_embedding(embedding_a).astype(np.float64, copy=False)
    vector_b = validate_embedding(embedding_b).astype(np.float64, copy=False)

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Embedding dimensions must match: "
            f"got {vector_a.shape[0]} and {vector_b.shape[0]}."
        )

    return vector_a, vector_b


def _validate_reference_and_batch(
    reference_embedding: Any, embeddings: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one reference embedding and a 2-D batch of embeddings."""

    reference = validate_embedding(reference_embedding).astype(np.float64, copy=False)

    try:
        batch = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embeddings batch must be numeric and convertible to float32.") from exc

    if batch.ndim != 2:
        raise ValueError("Embeddings batch must be a 2-D array with shape (n, d).")
    if batch.shape[0] == 0:
        raise ValueError("Embeddings batch must contain at least one embedding.")
    if batch.shape[1] == 0:
        raise ValueError("Embeddings in batch must not be empty.")
    if batch.shape[1] != reference.shape[0]:
        raise ValueError(
            "Batch embedding dimension must match reference dimension: "
            f"got {batch.shape[1]} and {reference.shape[0]}."
        )
    if not np.all(np.isfinite(batch)):
        raise ValueError("Embeddings batch must not contain NaN or infinite values.")

    return reference, batch.astype(np.float64, copy=False)
