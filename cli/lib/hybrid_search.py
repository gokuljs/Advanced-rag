import os

from .keyword_search import InvertedIndex
from .semantic_search import chunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = chunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
    
def normalized_score(scores):
    """
    Score min max normalization function.
    This is basically for hybrid search to combine the scores of the keyword search and the semantic search.
    this is 
    Args:
        scores (list): List of scores to normalize.
    Returns:
        list: Normalized scores.
    """
    if not scores or len(scores) == 0:
        return []
    minimumScore = min(scores)
    maximumScore = max(scores)
    score_range = maximumScore - minimumScore
    return [(score - minimumScore) / score_range for score in scores]