import argparse

from lib.hybrid_search import normalized_score

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    norm_passed_parser = subparsers.add_parser("normalize", help="Normalize a list of scores passed as arguments")
    norm_passed_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")
    

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print(normalized_score(args.scores))


if __name__ == "__main__":
    main()