import argparse
from lib.keyboard_search import search_command, build_command

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = search_command(args.query,5)
            for i, result in enumerate(result):
                print(f"{i+1}. {result['title']} \n")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()