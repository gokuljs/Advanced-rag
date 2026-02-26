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
from .search_utils import CACHE_PATH, load_movies, load_stopwords
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

    def get_documents(self, term):
        """
        Retrieve document IDs containing the specified term.
        
        Args:
            term: Search term to look up
            
        Returns:
            Sorted list of document IDs containing the term
        """
        return sorted(list(self.index[term]))
    
    def get_term_frequency(self, doc_id,term):
        """
        Retrieve the term frequency for a document.
        
        Args:
            doc_id: Document ID to look up
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        return self.term_frequency[doc_id][tokens[0]]

    def get_idf(self,term):
        """
        Retrieve the inverse document frequency for a term.
        
        Args:
            term: Term to look up
        """
        token= tokenize_text(term)
        if len(token) != 1:
            raise ValueError("can only have one token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log(doc_count + 1 / term_doc_count + 1)
        
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

def build_command():
    docs = InvertedIndex()
    docs.build()
    docs.save()


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
