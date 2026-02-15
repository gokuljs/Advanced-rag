"""
Keyword search module for movie search using inverted index.

This module provides functionality to build and query an inverted index
for fast keyword-based movie searches with text preprocessing and stemming.
"""

import os
import pickle
import string
from collections import defaultdict

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
        self.index_path = CACHE_PATH / "index"
        self.doc_path = CACHE_PATH / "docmap"

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

    def get_documents(self, term):
        """
        Retrieve document IDs containing the specified term.
        
        Args:
            term: Search term to look up
            
        Returns:
            Sorted list of document IDs containing the term
        """
        return sorted(list(self.index[term]))

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
    doc_ids = docs.get_documents("merida")
    print("Printing documents: ", doc_ids[0])


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
    results = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie["title"])
        if has_matching_tokens(query_tokens, movie_tokens):
            results.append(movie)
        if len(results) == n_results:
            break         
    return results
