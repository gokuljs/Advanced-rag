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
from .search_utils import load_movies
from .llm import correct_spelling, rewrite_query


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
        combinedResults = combine_search_results(bm25results, semanticResults)
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
        bm25results = self._bm25_search(query, limit * 500)
        semanticResults = self.semantic_search.search_chunks(query, limit * 500)
        combinedResults =  rrf_combine_search_results(bm25results, semanticResults,k)
        return combinedResults
    
def rrf_score(rank, k):
    return 1 / (k + rank)

def rrf_final_rank(bm25rank, sem_rank,k):
    if bm25rank and sem_rank:
        return rrf_score(bm25rank, k) + rrf_score(sem_rank, k)
    return 0
    

def rrf_combine_search_results(bm25results, semanticResults, k):
    """
    Recepiprocal rank fusion combine search results
    """
    scores ={}
    for rank,result in enumerate(bm25results,start=1):
        doc_id = result["doc_id"]
        scores[doc_id] = {
            "doc_id": doc_id,
            "bm25rank": rank,
            "bm25_score":rrf_score(rank, k),
            "sem_rank":None,
            "sem_score":None,
            "title":result["title"],
            "description":result["description"]   
        }
     
    for rank,result in enumerate(semanticResults, start=1):
        doc_id = result.get("doc_id") or result.get("id")
        if doc_id in scores:
            scores[doc_id]["sem_rank"] = rank
            scores[doc_id]["sem_score"] = rrf_score(rank, k)
        else:
            scores[doc_id] = {
                "doc_id": doc_id,
                "bm25rank": None,
                "bm25_score": None,
                "sem_rank": rank,
                "sem_score": rrf_score(rank, k),
                "title":result["title"],
                "description":result["description"]   
            }
    
    for doc_id in scores.keys():
        scores[doc_id]['rrf_score'] = rrf_final_rank(scores[doc_id]['bm25rank'], scores[doc_id]['sem_rank'],k)
        
    return sorted(list(scores.values()), key=lambda x: x['rrf_score'], reverse=True)
        

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
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


def normalize_scores(results):
    """
    Normalize a list of scores to a range of 0 to 1.
    """
    scores=[result["score"] for result in results]
    norm_scores=normalized_score(scores)
    for idx, result in enumerate(results):
        result["normalized_score"] = norm_scores[idx]
    return results

def combine_search_results(bm25results, semanticResults, alpha=0.5):
    """
    Combine and normalize results from BM25 and semantic search, then compute hybrid scores.

    This function takes two lists of search result dictionaries—one from BM25 retrieval
    and one from a semantic model. Each dictionary is expected to contain:
        - "doc_id": Unique identifier for the document.
        - "title": Title of the document.
        - "description": Description (or snippet) of the document.
        - "score": Raw relevance score from the respective method.

    The function normalizes both sets of scores to [0, 1], merges results on doc_id,
    and assigns missing scores for each method (BM25 or semantic) as zero if a document
    is only present in one list. For each document, it computes a hybrid score using
    `hybrid_score` and returns a list of merged results sorted in descending order of
    hybrid score.

    Args:
        bm25results (list[dict]): List of BM25 search result dictionaries.
        semanticResults (list[dict]): List of semantic search result dictionaries.

    Returns:
        list[dict]: Combined, normalized, and hybrid-scored result dictionaries with keys:
            - id
            - bm25_score
            - sem_score
            - title
            - description
            - hybrid_score
    """
    bm25_norm = normalize_scores(bm25results)
    sem_norm = normalize_scores(semanticResults)
    combinedNorm = {}
    for norm in bm25_norm:
        doc_id = norm["doc_id"]
        combinedNorm[doc_id] = {
            "id": doc_id,
            "bm25_score": norm["normalized_score"],
            "sem_score": 0,
            "title": norm["title"],
            "description": norm["description"],
        }
    for norm in sem_norm:
        # Semantic results use "id" while BM25 results use "doc_id"
        doc_id = norm.get("doc_id") or norm.get("id")
        if doc_id not in combinedNorm:
            combinedNorm[doc_id] = {
                "id": doc_id,
                "bm25_score": 0,
                "sem_score": norm["normalized_score"],
                "title": norm["title"],
                "description": norm["description"],
            }
        else:
            combinedNorm[doc_id]["sem_score"] = max(combinedNorm[doc_id]["sem_score"], norm["normalized_score"])

    for k, v in combinedNorm.items():
        combinedNorm[k]["hybrid_score"] = hybrid_score(v["bm25_score"], v["sem_score"], alpha)

    return sorted(combinedNorm.values(), key=lambda x: x["hybrid_score"], reverse=True)
    
        
    


def normalized_score(scores):
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

def weighted_search(query, alpha = 0.5, limit = 5):
    """
    Run a hybrid (weighted) search that combines BM25 and semantic scores for a given query,
    and print the top results.

    This function loads the movie documents, instantiates a HybridSearch object, executes the
    weighted hybrid search using the specified parameters, and prints nicely formatted
    information about each result.

    Args:
        query (str): The search query string.
        alpha (float): Weight for BM25 score when combining with semantic score (0.0 = pure semantic, 1.0 = pure BM25).
        limit (int): Maximum number of results to print.

    Prints:
        title, description (truncated to 1000 chars), hybrid score, BM25 score, and semantic score for each result.
    """
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.weighted_search(query, alpha, limit)
    for result in results[:limit]:
        print(f"Title: {result['title']}")
        print(f"Description: {result['description'][:1000]}")
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"BM25 Score: {result['bm25_score']}")
        print(f"Semantic Score: {result['sem_score']}")
        print("-" * 100)


def rrf_score_search(query, k=0.5, limit=5, enhance=None):
    """
    Run a Reciprocal Rank Fusion (RRF) hybrid search that merges BM25 and semantic search results,
    and print the top results.

    This function loads the movie documents, instantiates a HybridSearch object, applies optional
    query enhancement (currently, spelling correction), executes the RRF hybrid search using the 
    specified parameters, and prints nicely formatted information about each result.

    Args:
        query (str): The search query string.
        k (float): RRF hyperparameter controlling influence of rank positions (lower k → steeper drop-off).
        limit (int): Maximum number of results to print.
        enhance (str, optional): Optional query enhancement. If set to "spell", corrects the query's spelling.

    Prints:
        title, BM25 rank, semantic rank, and final RRF score for each result.
    """
    documents = load_movies()
    hs = HybridSearch(documents)
    match enhance:
        case "spell":
            new_query = correct_spelling(query)
            print(f"New query: {new_query} -> Original query: {query}")
            query = new_query
        case "rewrite":
            new_query = rewrite_query(query)
            print(f"New query: {new_query} -> Original query: {query}")
            query = new_query
    results = hs.rrf_search(query, k, limit)
    for result in results[:limit]:
        print(f"Title: {result['title']}")
        print(f"BM25 Rank: {result['bm25rank']}")
        print(f"Semantic Rank: {result['sem_rank']}")
        print(f"RRF Score: {result['rrf_score']}")
        print("-" * 100)