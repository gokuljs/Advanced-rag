"""
Hybrid search module combining BM25 keyword search and semantic (embedding-based) search.

Provides a ``HybridSearch`` class that initialises both an inverted index (BM25) and
a chunked semantic search engine, then merges their results using either a weighted
average or Reciprocal Rank Fusion (RRF) strategy.

Helper utilities for score normalisation and weighted combination are also exposed
here so callers can reuse them independently.
"""

import os

from .keyword_search import InvertedIndex
from .semantic_search import chunkedSemanticSearch


class HybridSearch:
    """
    Combines BM25 keyword search with dense semantic search over a document corpus.

    On construction the class immediately:
    * Loads (or builds) chunk-level sentence-transformer embeddings via
      ``chunkedSemanticSearch``.
    * Loads (or builds) the BM25 inverted index via ``InvertedIndex``.

    This means the first instantiation may be slow while caches are created;
    subsequent runs load from disk and are fast.

    Attributes:
        documents (list[dict]): The document corpus passed at construction time.
        semantic_search (chunkedSemanticSearch): Handles embedding-based retrieval.
        idx (InvertedIndex): Handles BM25-based retrieval.
    """

    def __init__(self, documents: list[dict]) -> None:
        """
        Initialise both search engines and warm up their caches.

        Args:
            documents: List of document dicts, each expected to contain at least
                ``id``, ``title``, and ``description`` keys.
        """
        self.documents = documents
        self.semantic_search = chunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        """
        Run a BM25 search against the inverted index.

        Loads the index from disk on every call (cheap pickle load) and delegates
        to ``InvertedIndex.bm25_search``.

        Args:
            query: Raw query string.  Tokenisation is handled internally.
            limit: Maximum number of results to return.

        Returns:
            List of result dicts ordered by descending BM25 score.  Each dict
            contains at least ``doc_id``, ``title``, and ``score``.
        """
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """
        Retrieve documents using a weighted combination of BM25 and semantic scores.

        Both retrievers are called with a large candidate pool (``limit * 500``) so
        that there is sufficient overlap to combine.  The raw scores from each
        retriever are normalised to [0, 1] via min-max normalisation before being
        merged with ``combine_results``.

        Args:
            query: Natural-language query string.
            alpha: Weight applied to the BM25 score.  The semantic score receives
                weight ``1 - alpha``.  A value of ``1.0`` is pure BM25; ``0.0`` is
                pure semantic.
            limit: Number of final results to return after merging. Defaults to 5.

        Returns:
            Merged and re-ranked list of result dicts.
        """
        bm25results = self._bm25_search(query, limit * 500)
        semanticResults = self.semantic_search.search_chunks(query, limit * 500)
        combinedResults = combine_results(bm25results, semanticResults)
        return combinedResults

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        """
        Retrieve documents using Reciprocal Rank Fusion (RRF).

        RRF combines ranked lists from multiple retrievers without needing score
        normalisation.  The fusion score for a document is the sum of
        ``1 / (k + rank_i)`` across all retrievers, where ``rank_i`` is its
        1-based position in retriever *i*'s result list.

        Args:
            query: Natural-language query string.
            k: Constant that dampens the impact of high-ranked documents.
                Typical values range from 10 to 60.
            limit: Number of final results to return. Defaults to 10.

        Raises:
            NotImplementedError: Always — this method has not been implemented yet.
        """
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    """
    Compute a single hybrid relevance score from BM25 and semantic sub-scores.

    Both input scores should be normalised to the same range (e.g. [0, 1]) before
    calling this function so that ``alpha`` acts as an interpretable mixing weight.

    Args:
        bm25_score: Normalised BM25 (keyword) relevance score for a document.
        semantic_score: Normalised semantic (embedding) relevance score for a document.
        alpha: Mixing weight for BM25.  Must be in [0, 1].  The semantic score
            receives weight ``1 - alpha``.

    Returns:
        Weighted average of the two scores.
    """
    return bm25_score * alpha + semantic_score * (1 - alpha)


def normalized_score(scores: list[float]) -> list[float]:
    """
    Apply min-max normalisation to a list of scores, mapping them to [0, 1].

    Used during hybrid search to bring BM25 and semantic scores onto a common
    scale before combining them with ``hybrid_score``.

    Args:
        scores: List of raw numeric scores to normalise.  May contain any finite
            float values.

    Returns:
        List of normalised scores in [0, 1].  Returns an empty list if the input
        is empty.  If all scores are identical (zero range) every output value will
        be ``0.0`` due to division by zero protection.
    """
    if not scores or len(scores) == 0:
        return []
    minimumScore = min(scores)
    maximumScore = max(scores)
    score_range = maximumScore - minimumScore
    if score_range == 0:
        return [0.0 for _ in scores]
    return [(score - minimumScore) / score_range for score in scores]
