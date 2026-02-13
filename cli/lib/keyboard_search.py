import string
from .search_utils import load_movies, load_stopwords

def transform_text(str):
    text = str.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text):
    text = transform_text(text)
    stopwords = load_stopwords()
    def _filter(tok):
        if tok and tok not in stopwords:
            return True 
        return False
    token = [tok for tok in text.split() if _filter(tok)]
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
