import argparse
from lib.keyboard_search import search_command, build_command, tf_command, idf_command

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser= subparsers.add_parser("tf", help="Term frequency")
    search_parser.add_argument("doc_id", type=int, help="Document ID")
    search_parser.add_argument("term", type=str, help="Term")
    search_parser= subparsers.add_parser("idf", help="Inverse document frequency")
    search_parser.add_argument("term", type=str, help="Term")
    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = search_command(args.query,5)
            for i, result in enumerate(result):
                print(f"{i+1}. {result['title']} \n")
        case "build":
            build_command()
        case "tf":
            print(f"Term frequency for {args.term} in document {args.doc_id}")
            result = tf_command(args.doc_id,args.term)
            print(result)
        case "idf":
            print(f"Inverse document frequency for {args.term}")
            result = idf_command(args.term)
            print(result)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()