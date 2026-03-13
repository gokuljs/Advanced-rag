"""
Command-line interface for hybrid search utilities.

Exposes two sub-commands:

* ``normalize`` – apply min-max normalisation to a list of raw scores passed
  directly as CLI arguments.  Useful for quick manual testing of the score
  normalisation step without running a full search.
* ``weightedsearch`` – run a hybrid BM25 + semantic search with a tunable
  alpha mixing weight and return the top results.

Usage examples::

    python Hybrid_search_cli.py normalize 0.2 0.5 0.9 1.3
    python Hybrid_search_cli.py weightedsearch "space western" 0.4 10
"""

import argparse

from lib.hybrid_search import normalized_score


def main() -> None:
    """
    Parse CLI arguments and dispatch to the appropriate hybrid-search command.

    Sub-commands
    ------------
    normalize
        Accepts one or more float scores as positional arguments and prints
        the min-max normalised equivalents as a Python list.

    weightedsearch
        Accepts a query string, an alpha weight (BM25 contribution), and an
        optional result limit, then runs a weighted hybrid search.  The
        ``weightedsearch`` handler is currently wired up in the argument parser
        but not yet dispatched in the ``match`` block — it is a placeholder for
        future implementation.
    """
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_passed_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores passed as arguments",
    )
    norm_passed_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="List of scores to normalize",
    )

    ws_parser = subparsers.add_parser(
        "weightedsearch",
        help="A weighted search is hybrid search with weighted average combination",
    )
    ws_parser.add_argument("query", type=str, help="Query text to search for")
    ws_parser.add_argument(
        "alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 score (0.0 = pure semantic, 1.0 = pure BM25)",
    )
    ws_parser.add_argument(
        "limit",
        type=int,
        default=5,
        help="Maximum number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print(normalized_score(args.scores))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
