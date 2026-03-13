"""
Semantic search module using sentence-transformer embeddings.

Provides two search classes:

* ``SemanticSearch`` – encodes full document texts into dense vectors and
  retrieves the most similar ones via cosine similarity.
* ``chunkedSemanticSearch`` – splits each document description into overlapping
  sentence-level chunks before encoding, then aggregates per-document scores
  by taking the maximum chunk similarity.  This improves recall for long texts
  where only a portion is relevant to the query.

Standalone helper functions for chunking, embedding, and running searches from
the CLI are also included.
"""

from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from .search_utils import load_movies
import json

CACHE_PATH = Path(__file__).resolve().parents[1] / "cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)


class SemanticSearch:
    """
    Dense retrieval engine that encodes full document texts as sentence-transformer
    embeddings and ranks them by cosine similarity to a query embedding.

    Embeddings are cached to disk as a ``.npy`` file so they are only computed
    once.  On subsequent runs the cache is validated by comparing its length to
    the current document count; a mismatch triggers a rebuild.

    Attributes:
        model (SentenceTransformer): Loaded sentence-transformer model.
        embeddings_path (Path): Filesystem path where embeddings are cached.
        embeddings (np.ndarray | None): Matrix of shape (N, D) after loading.
        documents (list[dict]): The indexed document corpus.
        document_map (dict): Mapping of ``doc["id"]`` → document dict for O(1) lookups.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialise the search engine with a sentence-transformer model.

        Args:
            model_name: HuggingFace model identifier accepted by
                ``SentenceTransformer``.  Defaults to ``"all-MiniLM-L6-v2"``,
                a lightweight 384-dimensional model that balances speed and quality.
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = CACHE_PATH / "embeddings.npy"
        self.embeddings = None
        self.documents = []
        self.document_map = {}

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Encode all documents and persist the resulting embeddings to disk.

        Each document is encoded as the concatenation of its ``title`` and
        ``description`` fields.  The embedding matrix is saved to
        ``self.embeddings_path`` so future calls can skip re-encoding.

        Args:
            documents: List of document dicts.  Each dict must contain ``id``,
                ``title``, and ``description`` keys.

        Returns:
            NumPy array of shape ``(len(documents), embedding_dim)``.
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

    def load_and_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Load embeddings from the on-disk cache, rebuilding if the cache is stale.

        The cache is considered valid when the number of stored embeddings equals
        the number of documents.  If the cache is missing or the document count
        has changed, ``build_embeddings`` is called to regenerate it.

        Args:
            documents: List of document dicts (same format as ``build_embeddings``).

        Returns:
            NumPy array of shape ``(len(documents), embedding_dim)``.
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

    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Encode a single piece of text into a dense embedding vector.

        Args:
            text: Non-empty string to encode.

        Returns:
            1-D NumPy array of shape ``(embedding_dim,)``.

        Raises:
            ValueError: If ``text`` is ``None`` or consists only of whitespace.
        """
        if text is None or text.strip() == "":
            raise ValueError("Text cannot be None or empty")
        return self.model.encode([text])[0]

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Return the top-``n_results`` documents most similar to ``query``.

        Computes cosine similarity between the query embedding and every document
        embedding, then returns the highest-scoring documents in descending order.

        Args:
            query: Natural-language query string.
            n_results: Maximum number of results to return.  Defaults to 10.

        Returns:
            List of dicts, each containing ``score``, ``title``, and
            ``description``, ordered by descending similarity.

        Raises:
            ValueError: If ``load_and_create_embeddings`` (or ``build_embeddings``)
                has not been called first.
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


class chunkedSemanticSearch(SemanticSearch):
    """
    Semantic search that operates at the sentence-chunk level rather than the full
    document level.

    Each document description is split into overlapping sentence chunks via
    ``semantic_chunking`` before encoding.  During search, every chunk of every
    document is scored against the query and the document's final relevance score
    is the maximum chunk score.  This gives better recall for long documents where
    only a paragraph is relevant.

    Chunk embeddings and their metadata are cached to disk independently of the
    full-document embeddings used by the parent ``SemanticSearch`` class.

    Class Attributes:
        chunk_embeddings_path (Path): Cache file for the chunk embedding matrix.
        chunk_metadata_path (Path): JSON file mapping each chunk to its source document.
    """

    chunk_embeddings_path: Path
    chunk_metadata_path: Path

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialise the chunked search engine.

        Args:
            model_name: HuggingFace model identifier forwarded to the parent class.
                Defaults to ``"all-MiniLM-L6-v2"``.
        """
        super().__init__(model_name)
        self.chunk_embeddings_path = CACHE_PATH / "chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH / "chunk_metadata.json"
        self.chunk_size = None
        self.overlap = None

    def build_chunked_embeddings(self, document: list[dict]) -> np.ndarray:
        """
        Chunk every document description, encode all chunks, and persist the results.

        Uses ``semantic_chunking`` with ``overlap=1`` and ``max_chunk_size=4``
        (i.e. up to 4 sentences per chunk, 1-sentence overlap between consecutive
        chunks).  Documents with empty descriptions are skipped.

        The chunk embeddings are saved as a ``.npy`` matrix and the per-chunk
        metadata (source document index, chunk index, total chunk count) is saved
        as a JSON file.

        Args:
            document: List of document dicts with ``id``, ``title``, and
                ``description`` keys.

        Returns:
            NumPy array of shape ``(total_chunks, embedding_dim)``.
        """
        self.documents = document
        self.document_map = {doc["id"]: doc for doc in self.documents}
        all_chunks = []
        chunk_metadata = []

        for midx, doc in enumerate(document):
            if doc["description"] is None or doc["description"].strip() == "":
                continue
            chunks = semantic_chunking(doc["description"], overlap=1, max_chunk_size=4)
            all_chunks.extend(chunks)
            for cidx in range(len(chunks)):
                chunk_metadata.append(
                    {
                        "movie_idx": midx,
                        "chunk_idx": cidx,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Load chunk embeddings from the on-disk cache, building them if absent.

        Unlike the parent class's ``load_and_create_embeddings``, this method does
        not validate cache freshness beyond checking whether both cache files exist.
        Re-run ``build_chunked_embeddings`` manually if the document corpus changes.

        Args:
            documents: List of document dicts (same format as
                ``build_chunked_embeddings``).

        Returns:
            NumPy array of shape ``(total_chunks, embedding_dim)``.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in self.documents}
        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunked_embeddings(self.documents)

    def search_chunks(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Return the top-``n_results`` documents most relevant to ``query``.

        Scores every chunk embedding against the query embedding using cosine
        similarity, then assigns each document the maximum score across all its
        chunks.  This "max-pooling" strategy ensures that a document is surfaced
        if *any* part of it is highly relevant, even if the rest is not.

        Args:
            query: Natural-language query string.
            n_results: Maximum number of documents to return. Defaults to 10.

        Returns:
            List of dicts ordered by descending relevance, each containing:

            * ``id`` – document ID.
            * ``title`` – document title.
            * ``description`` – first 100 characters of the description.
            * ``score`` – maximum chunk cosine similarity (float in [-1, 1]).
            * ``metadata`` – chunk metadata dict from the JSON cache.
        """
        query_embedding = self.generate_embeddings(query)
        chunk_score = []
        movie_score = defaultdict(lambda: 0)
        for idx, chunk_emb in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata["chunks"][idx]
            midx, cidx = metadata["movie_idx"], metadata["chunk_idx"]
            sim = cosine_similarity(query_embedding, chunk_emb)
            chunk_score.append({"score": sim, "movie_idx": midx, "chunk_idx": cidx})
            movie_score[midx] = max(movie_score[midx], sim)
        movie_score_sorted = sorted(movie_score.items(), key=lambda x: x[1], reverse=True)
        res=[]
        for movie_idx, score in movie_score_sorted[:n_results]:
            res.append({
                "id": self.documents[movie_idx]["id"],
                "title": self.documents[movie_idx]["title"],
                "description": self.documents[movie_idx]["description"][:100],
                "score": score,
                "metadata": self.chunk_metadata["chunks"][movie_idx] or {},
            })
        return res


def verify_model(model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Load a sentence-transformer model and print basic diagnostics.

    Useful as a quick sanity check to confirm that the model weights are
    accessible and that the maximum sequence length is as expected.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``"all-MiniLM-L6-v2"``.
    """
    model = SentenceTransformer(model_name)
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str) -> np.ndarray:
    """
    Convenience wrapper: encode a single string using a default ``SemanticSearch`` instance.

    Args:
        text: Non-empty string to encode.

    Returns:
        1-D NumPy embedding vector.
    """
    return SemanticSearch().generate_embeddings(text)


def verify_embeddings() -> None:
    """
    Load (or build) full-document embeddings and print shape diagnostics.

    Loads the movie corpus, calls ``load_and_create_embeddings``, then prints
    the document count and the number of cached embedding rows to confirm they
    match.  Intended as a development/debugging helper.
    """
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_and_create_embeddings(documents)
    print("length of the documents: ", len(documents))
    print("embedding shape: ", embeddings.shape[0])


def embed_query_text(query: str) -> None:
    """
    Encode a query string and print its shape and raw values.

    Intended as a development/debugging helper to inspect what the embedding
    looks like for a given piece of text.

    Args:
        query: Query string to encode.
    """
    ss = SemanticSearch()
    result = ss.generate_embeddings(query)
    print(query)
    print("embedding shape: ", result.shape)
    print("embedding: ", result)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two dense vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    giving a value in [-1, 1] where 1 means identical direction and -1 means
    opposite direction.  It is used here to compare query and document embeddings
    independently of their magnitudes.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector (must have the same dimensionality as ``vec1``).

    Returns:
        Cosine similarity as a float in [-1, 1].  Returns ``0.0`` if either
        vector is the zero vector (to avoid division by zero).
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query: str, n_results: int = 5) -> None:
    """
    Run a full-document semantic search and print the results to stdout.

    Loads the movie corpus, builds/loads full-document embeddings, runs
    ``SemanticSearch.search``, and prints each result's rank, title,
    description snippet (up to 1 000 chars), and cosine similarity score.

    Args:
        query: Natural-language query string.
        n_results: Number of results to display.  Defaults to 5.
    """
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_and_create_embeddings(documents)
    results = ss.search(query, n_results)
    for i, result in enumerate(results):
        print(
            f"{i + 1}. {result['title']} \n {result['description'].strip()[0:1000]} \n Score: {result['score']}"
        )


def fixed_size_chunking(text: str, overlap: int, chunk_size: int = 1000) -> list[str]:
    """
    Split text into fixed-size word chunks with optional overlap.

    Splits on whitespace and groups ``chunk_size`` words per chunk, advancing
    by ``chunk_size - overlap`` words between consecutive chunks.  Trailing
    chunks shorter than ``overlap`` words are discarded to avoid producing
    near-empty chunks.

    Args:
        text: Input text to chunk.
        overlap: Number of words shared between consecutive chunks.  Must be
            strictly less than ``chunk_size``.
        chunk_size: Number of words per chunk.  Defaults to 1 000.

    Returns:
        List of chunk strings.

    Raises:
        ValueError: If ``overlap >= chunk_size``, which would cause a
            non-positive step size.
    """
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be less than chunk_size (otherwise step size would be zero)"
        )
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def semantic_chunking(text, overlap=0, max_chunk_size=4):
    """
    Chunk text on sentence boundaries to preserve meaning.
    Splits on sentence endings (. ! ?) and groups up to max_chunk_size sentences per chunk,
    with optional overlap of sentences between consecutive chunks.
    """
    text = text.strip()
    if text is None or text.strip() == "":
        return []
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max_chunk_size")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if not chunk_sentences:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


def semantic_chunk_command(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    """
    Run ``semantic_chunking`` on ``text`` and print each chunk to stdout.

    Prints a summary line with the total chunk count, then each chunk prefixed
    by its 1-based index.  Intended as a CLI debugging helper.

    Args:
        text: Text to chunk.
        max_chunk_size: Maximum number of sentences per chunk.  Defaults to 4.
        overlap: Number of sentences shared between consecutive chunks.  Defaults to 0.
    """
    chunks = semantic_chunking(text, overlap=overlap, max_chunk_size=max_chunk_size)
    print(
        f"Semantic chunking: {len(chunks)} chunks (max {max_chunk_size} sentences, overlap {overlap})"
    )
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def chunk_text(query: str, overlap: int, chunk_size: int = 200) -> None:
    """
    Run ``fixed_size_chunking`` and print each chunk to stdout.

    Prints a summary with the total chunk count, then each chunk prefixed by
    its 1-based index.  Intended as a CLI debugging helper.

    Args:
        query: Text to chunk.
        overlap: Number of words shared between consecutive chunks.
        chunk_size: Number of words per chunk.  Defaults to 200.
    """
    chunks = fixed_size_chunking(query, overlap, chunk_size)
    print(f"chunking {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks() -> None:
    """
    Load (or build) chunk-level embeddings for the movie corpus and print diagnostics.

    Loads all movies, calls ``chunkedSemanticSearch.load_or_create_chunk_embeddings``,
    and prints the resulting embedding array.  Intended as a development/debugging
    helper to confirm that the chunk cache is populated correctly.
    """
    movies = load_movies()
    css = chunkedSemanticSearch()
    chunk_embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"length of the chunked embeddings: {chunk_embeddings}")


def search_chunks_command(query: str, n_results: int = 5) -> None:
    """
    Run a chunked semantic search and print the top results to stdout.

    Loads the movie corpus, builds/loads chunk embeddings, runs
    ``chunkedSemanticSearch.search_chunks``, and prints each result's rank,
    title, description snippet (up to 1 000 chars), and max-chunk similarity
    score.

    Args:
        query: Natural-language query string.
        n_results: Number of results to display.  Defaults to 5.
    """
    css = chunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, n_results)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['title']} \n {result['description'].strip()[0:1000]} \n Score: {result['score']}")