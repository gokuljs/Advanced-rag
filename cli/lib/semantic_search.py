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

