import argparse
from lib.keyboard_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    get_tf_idf_command,
    get_bm25_idf_command,
    get_bm25_tf_command,
    bm25_search_command,
)
from lib.search_utils import BM25_K1, BM25_B

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("tf", help="Term frequency")
    search_parser.add_argument("doc_id", type=int, help="Document ID")
    search_parser.add_argument("term", type=str, help="Term")
    search_parser = subparsers.add_parser("idf", help="Inverse document frequency")
    search_parser.add_argument("term", type=str, help="Term")
    search_parser = subparsers.add_parser("tfidf", help="TF-IDF")
    search_parser.add_argument("doc_id", type=int, help="Document ID")
    search_parser.add_argument("term", type=str, help="Term")
    search_parser = subparsers.add_parser("bm25idf", help="BM25 IDF")
    search_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter"
    )
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()
    match args.command:
        case "bm25search":
            result = bm25_search_command(args.query, 5)
            for i, result in enumerate(result):
                print(f"{i}. '{result['doc_id']}' {result['title']} {result['score']}")
        case "search":
            print(f"Searching for: {args.query}")
            result = search_command(args.query, 5)
            for i, result in enumerate(result):
                print(f"{i + 1}. {result['title']} \n")
        case "build":
            build_command()
        case "tf":
            print(f"Term frequency for {args.term} in document {args.doc_id}")
            result = tf_command(args.doc_id, args.term)
            print(result)
        case "idf":
            print(f"Inverse document frequency for {args.term}")
            result = idf_command(args.term)
            print(result)
        case "tfidf":
            print(f"TF-IDF for {args.term} in document {args.doc_id}")
            result = get_tf_idf_command(args.doc_id, args.term)
            print(result)
        case "bm25idf":
            result = get_bm25_idf_command(args.term)
            print(result)
        case "bm25tf":
            result = get_bm25_tf_command(args.doc_id, args.term, BM25_K1, BM25_B)
            print(result)
        case "bm25search":
            result = bm25_search_command(args.query, 5)
            for i, result in enumerate(result):
                print(f"{i + 1}. {result['title']} \n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
