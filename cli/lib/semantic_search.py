from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from .search_utils import load_movies

CACHE_PATH = Path(__file__).resolve().parents[1] / "cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings_path = CACHE_PATH / "embeddings.npy"
        self.embeddings = None
        self.documents = []
        self.document_map = {}

    def build_embeddings(self, documents):
        """
        Build embeddings for a given list of documents.
        """
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for movie in self.documents:
            self.document_map[movie["id"]] = movie
            movie_strings.append(f"{movie['title']} {movie['description']}")
        self.embeddings = self.model.encode(movie_strings)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_and_create_embeddings(self, documents):
        """
        Load embeddings from cache and create embeddings for a given list of documents.
        """
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(self.documents)

    def generate_embeddings(self, text: str):
        """
        Generate embeddings for a given text.
        """
        if text is None or text.strip() == "":
            raise ValueError("Text cannot be None or empty")
        return self.model.encode([text])[0]

    def search(self, query: str, n_results: int = 10):
        """
        Search for the most similar documents to a given query.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings are not loaded")
        query_embedding = self.generate_embeddings(query)
        similarities = []
        for doce_emb, doc in zip(self.embeddings, self.documents):
            similarity = cosine_similarity(query_embedding, doce_emb)
            similarities.append((similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": sc, "title": doc["title"], "description": doc["description"]}
            for sc, doc in similarities[:n_results]
        ]


def verify_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Verify the model is loaded correctly.
    """
    model = SentenceTransformer(model_name)
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str):
    """
    Embed a given text.
    """
    return SemanticSearch().generate_embeddings(text)


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_and_create_embeddings(documents)
    print("length of the documents: ", len(documents))
    print("embedding shape: ", embeddings.shape[0])


def embed_query_text(query):
    """
    Embed a given query text.
    """
    ss = SemanticSearch()
    result = ss.generate_embeddings(query)
    print(query)
    print("embedding shape: ", result.shape)
    print("embedding: ", result)


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query, n_results=5):
    """
    Search for the most similar documents to a given query.
    """
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_and_create_embeddings(documents)
    results = ss.search(query, n_results)
    for i,result in enumerate(results):
        print(f"{i + 1}. {result['title']} \n {result['description'].strip()[0:100]} \n Score: {result['score']}")
