"""
Keyword search module for movie search using inverted index.

This module provides functionality to build and query an inverted index
for fast keyword-based movie searches with text preprocessing and stemming.
"""

import os
import pickle
import string
import math
from collections import defaultdict
from collections import Counter
from nltk.stem import PorterStemmer
from .search_utils import CACHE_PATH, load_movies, load_stopwords, BM25_K1, BM25_B
stemmer = PorterStemmer()


class InvertedIndex:
    """
    Inverted index data structure for efficient text search.
    
    Maps tokens to sets of document IDs containing those tokens,
    enabling fast keyword lookups across a collection of documents.
    """
    
    def __init__(self):
        """Initialize an empty inverted index with document mapping."""
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequency = defaultdict(Counter)
        self.index_path = CACHE_PATH / "index.pkl"
        self.doc_path = CACHE_PATH / "docmap.pkl"
        self.term_frequency_path = CACHE_PATH / "term_frequency.pkl"
        self.doc_lengths = {}
        self.doc_length_path = CACHE_PATH / "doc_lengths.pkl"

    def _add_document(self, doc_id, text):
        """
        Add a document to the inverted index.
        
        Args:
            doc_id: Unique identifier for the document
            text: Text content to be indexed
        """
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequency[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        
    def get_avg_doc_length(self) -> float:
        """
        Compute the average token count across all indexed documents.

        Used by ``get_bm25_tf`` as the length-normalisation denominator.
        A higher value causes BM25 to penalise long documents more aggressively.

        Returns:
            Mean document length in tokens, or ``0.0`` if no documents have
            been indexed yet.
        """
        lengths = list(self.doc_lengths.values())
        if len(lengths) == 0:
            return 0.0
        ttl = 0 
        for length in lengths: 
            ttl+=1
        return ttl / len(lengths)
    
    

    def get_documents(self, term):
        """
        Retrieve document IDs containing the specified term.
        
        Args:
            term: Search term to look up
            
        Returns:
            Sorted list of document IDs containing the term
        """
        return sorted(list(self.index[term]))
    
    def get_term_frequency(self, doc_id: int, term: str) -> int:
        """
        Return how many times ``term`` appears in document ``doc_id``.

        The term is passed through the same tokenisation pipeline used at index
        time (lowercasing, punctuation removal, stopword filtering, stemming) so
        that raw surface forms (e.g. ``"running"``) resolve to their stem
        (e.g. ``"run"``) correctly.

        Args:
            doc_id: Unique identifier of the document to query.
            term: Surface-form term (pre-tokenisation).  Must resolve to exactly
                one token after processing.

        Returns:
            Integer count of how many times the stemmed form of ``term`` occurs
            in the document's token list.

        Raises:
            ValueError: If ``term`` tokenises to more than one token, which would
                make the lookup ambiguous.
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        return self.term_frequency[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        """
        Compute the smoothed IDF weight for a term using the classic TF-IDF formula.

        Uses the add-1 smoothed variant:
        ``IDF(t) = log((N + 1) / (df(t) + 1))``
        where *N* is the total number of documents and *df(t)* is the number of
        documents containing the term.  The +1 smoothing prevents division by zero
        for unseen terms and dampens the influence of very common ones.

        Args:
            term: Surface-form term.  Must tokenise to exactly one token.

        Returns:
            Non-negative float IDF weight.  Higher values indicate rarer terms.

        Raises:
            ValueError: If ``term`` tokenises to more than one token.
        """
        token= tokenize_text(term)
        if len(token) != 1:
            raise ValueError("can only have one token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        """
        Compute the classic TF-IDF relevance score for a term in a document.

        TF-IDF = TF(term, doc) × IDF(term).  Unlike BM25, the raw term frequency
        is used without length normalisation.

        Args:
            doc_id: Unique identifier of the document.
            term: Surface-form term (must tokenise to exactly one token).

        Returns:
            Float TF-IDF score.
        """
        tf = self.get_term_frequency(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def bm25_search(self, query: str, n_results: int) -> list[dict]:
        """
        Rank all documents against ``query`` using the BM25 scoring function.

        Tokenises the query, sums the BM25 TF-IDF contribution of each query
        token across every document in the corpus, and returns the top-ranked
        documents in descending score order.

        Args:
            query: Raw natural-language query string.
            n_results: Maximum number of results to return.

        Returns:
            List of dicts (up to ``n_results``) each containing:

            * ``doc_id`` – document identifier.
            * ``title`` – document title.
            * ``score`` – aggregated BM25 score (higher is more relevant).
        """
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score =0
            for token in tokens:
                score += self.get_bm25_tfidf(doc_id, token)
            scores[doc_id] = score
        document_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = document_scores[:n_results]
        format_results=[]
        for doc_id, score in result:
            doc=self.docmap[doc_id]
            format_results.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "score": score
            })
            
        return format_results
    
    def get_bm25_tfidf(self, doc_id: int, term: str) -> float:
        """
        Compute the full BM25 relevance score for a single term in a document.

        Combines the BM25 TF component (length-normalised) with the BM25 IDF
        component:  ``BM25(t, d) = BM25_TF(t, d) × BM25_IDF(t)``.

        Args:
            doc_id: Unique identifier of the document.
            term: Surface-form term (must tokenise to exactly one token).

        Returns:
            Float BM25 score for this term–document pair.
        """
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """
        Compute the BM25 term-frequency component for a term in a document.

        Uses the standard BM25 saturation formula with length normalisation:
        ``BM25_TF = (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × (|d| / avgdl)))``

        where ``|d|`` is the document length in tokens and ``avgdl`` is the
        corpus-wide average document length.

        Args:
            doc_id: Unique identifier of the document.
            term: Surface-form term (must tokenise to exactly one token).
            k1: Controls term-frequency saturation.  Higher values reduce
                saturation, giving more weight to repeated terms.
                Defaults to ``BM25_K1`` (1.5).
            b: Controls the degree of document-length normalisation.  ``b=1``
                is full normalisation; ``b=0`` disables it.
                Defaults to ``BM25_B`` (0.75).

        Returns:
            Float BM25 TF component for this term–document pair.
        """
        tf = self.get_term_frequency(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.get_avg_doc_length()
        idf = self.get_idf(term)
        if(avg_doc_length >0):
            length_norm = 1 - b +b * (doc_length / avg_doc_length)
        else:
            length_norm = 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_bm25_idf(self, term: str) -> float:
        """
        Compute the Robertson–Spärck Jones BM25 IDF weight for a term.

        Uses the probabilistic BM25 IDF variant:
        ``BM25_IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)``

        This differs from the classic TF-IDF IDF in that it explicitly models
        the probability of relevance.  The ``+ 1`` inside the log keeps the
        result positive even for very common terms.

        Args:
            term: Surface-form term (must tokenise to exactly one token).

        Returns:
            Non-negative float BM25 IDF weight.

        Raises:
            ValueError: If ``term`` tokenises to more than one token.
        """
        token= tokenize_text(term)
        if len(token) != 1:
            raise ValueError("can only have one token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        
    def build(self):
        """
        Build the inverted index from the movie dataset.
        
        Loads all movies and indexes their title and description fields.
        """
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, text)
            self.docmap[doc_id] = movie
    

    def save(self):
        """
        Persist the inverted index and document map to disk.
        
        Saves both the index and docmap as pickle files in the cache directory.
        """
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.doc_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequency_path, "wb") as f:
            pickle.dump(self.term_frequency, f) 
        with open(self.doc_length_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
            
    def load(self):
        """
        Load the inverted index and document map from disk.
        
        Loads both the index and docmap from pickle files in the cache directory.
        """
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.doc_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequency_path, "rb") as f:
            self.term_frequency = pickle.load(f)
        with open(self.doc_length_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
    
            


def transform_text(text):
    """
    Normalize text by converting to lowercase and removing punctuation.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tf_command(doc_id, term):
    """
    Retrieve the term frequency for a specific term in a given document.

    This function loads the inverted index from disk and returns the frequency
    of the provided term within the specified document.

    Args:
        doc_id (int): The unique identifier of the document.
        term (str): The term whose frequency is to be retrieved.

    Returns:
        int: Frequency of the term in the given document.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_term_frequency(doc_id, term)

def idf_command(term):
    """
    Retrieve the inverse document frequency (IDF) for a specific term.

    This function loads the inverted index from disk and calculates the IDF
    value for the given term across the index's corpus.

    Args:
        term (str): The term whose IDF is to be computed.

    Returns:
        float: The inverse document frequency of the given term.
    """
    idx = InvertedIndex()
    idx.load()
    print("Print inverse document frequency for the term: ", term, "is", idx.get_idf(term))
    return idx.get_idf(term)

def get_bm25_idf_command(term: str) -> float:
    """
    Load the index from disk, compute the BM25 IDF for ``term``, and print it.

    Args:
        term: Surface-form term whose BM25 IDF is to be computed.

    Returns:
        Float BM25 IDF weight for the term.
    """
    idx = InvertedIndex()
    idx.load()
    print("Print BM25 IDF for the term: ", term, "is", idx.get_bm25_idf(term))
    return idx.get_bm25_idf(term)

def get_tf_idf_command(doc_id: int, term: str) -> float:
    """
    Load the index from disk, compute TF-IDF for ``term`` in ``doc_id``, and print it.

    Args:
        doc_id: Unique identifier of the document.
        term: Surface-form term (must tokenise to exactly one token).

    Returns:
        Float TF-IDF score for the term–document pair.
    """
    idx = InvertedIndex()
    idx.load()
    print("Print TF-IDF for the term: ", term, "in document: ", doc_id, "is", idx.get_tf_idf(doc_id, term))
    return idx.get_tf_idf(doc_id, term)

def get_bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """
    Load the index from disk, compute the BM25 TF component for ``term`` in ``doc_id``, and print it.

    Args:
        doc_id: Unique identifier of the document.
        term: Surface-form term (must tokenise to exactly one token).
        k1: BM25 saturation parameter.  Defaults to ``BM25_K1`` (1.5).
        b: BM25 length-normalisation parameter.  Defaults to ``BM25_B`` (0.75).

    Returns:
        Float BM25 TF component for the term–document pair.
    """
    idx = InvertedIndex()
    idx.load()
    print("Print BM25 TF for the term: ", term, "in document: ", doc_id, "is", idx.get_bm25_tf(doc_id, term, k1, b))
    return idx.get_bm25_tf(doc_id, term, k1, b)
     

def tokenize_text(text):
    """
    Tokenize and process text for indexing and searching.
    
    Performs the following operations:
    1. Text normalization (lowercase, punctuation removal)
    2. Stopword filtering
    3. Stemming using Porter Stemmer
    
    Args:
        text: Input text string
        
    Returns:
        List of processed tokens
    """
    text = transform_text(text)
    stopwords = load_stopwords()

    def _filter(tok):
        """Filter out empty tokens and stopwords."""
        return tok and tok not in stopwords

    tokens = [tok for tok in text.split() if _filter(tok)]
    tokens = [stemmer.stem(tok) for tok in tokens]
    return tokens


def has_matching_tokens(query_tokens, movie_tokens):
    """
    Check if any query token matches any movie token.
    
    Uses substring matching to allow partial word matches.
    
    Args:
        query_tokens: List of tokens from the search query
        movie_tokens: List of tokens from the movie text
        
    Returns:
        True if any query token is found in any movie token, False otherwise
    """
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False

def build_command() -> None:
    """
    Build the inverted index from the movie dataset and persist it to disk.

    Constructs a fresh ``InvertedIndex``, indexes every movie's title and
    description, then serialises the index, document map, term frequencies,
    and document lengths to pickle files in the cache directory.

    This must be run at least once before any search commands can be used.
    """
    docs = InvertedIndex()
    docs.build()
    docs.save()


def bm25_search_command(query: str, n_results: int = 5) -> list[dict]:
    """
    Load the inverted index from disk and run a BM25 search.

    Convenience wrapper around ``InvertedIndex.bm25_search`` intended for use
    by CLI entry points.

    Args:
        query: Raw natural-language query string.
        n_results: Maximum number of results to return.  Defaults to 5.

    Returns:
        List of result dicts containing ``doc_id``, ``title``, and ``score``,
        ordered by descending BM25 score.
    """
    docs = InvertedIndex()
    docs.load()
    result = docs.bm25_search(query, n_results)
    return result

def search_command(query, n_results):
    """
    Search for movies matching the query string.
    
    Performs a simple keyword search by tokenizing the query and
    matching against movie titles.
    
    Args:
        query: Search query string
        n_results: Maximum number of results to return
        
    Returns:
        List of movie dictionaries matching the query (up to n_results)
    """
    movies = load_movies()
    index = InvertedIndex()
    index.load()
    seen,results = set(),[]
    query_tokens = tokenize_text(query)
    for query in query_tokens:
        mathcing_doc_ids = index.get_documents(query)
        for doc_id in mathcing_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(index.docmap[doc_id])
            if len(results) == n_results:
                return results
    return results
