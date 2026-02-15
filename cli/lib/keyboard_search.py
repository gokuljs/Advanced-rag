from ntpath import exists
import string
import os
import pickle
from cli.lib.search_utils import CACHE_PATH, load_movies, load_stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH / "index"
        self.doc_path = CACHE_PATH / "docmap"

    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_document(self, term):
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, text)
            self.docmap[doc_id] = movie


    def save(self):
        os.mkdir(CACHE_PATH,exists_ok=True)
        with open(self.index_path, "w") as f:
            pickle.dump(self.index, f)
        with open(self.doc_path, "w") as f:
            pickle.dump(self.docmap, f)


def transform_text(str):
    text = str.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text):
    text = transform_text(text)
    stopwords = load_stopwords()

    def _filter(tok):
        if tok and tok not in stopwords:
            return True
        return False

    token = [tok for tok in text.split() if _filter(tok)]
    token = [stemmer.stem(tok) for tok in token]
    return token


def has_matching_tokens(query_tokens, movie_tokens):
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False


def search_command(query, n_results):
    movies = load_movies()
    res = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie["title"])
        if has_matching_tokens(query_tokens, movie_tokens):
            res.append(movie)
        if len(res) == n_results:
            break
    return res
