import argparse
from lib.semantic_search import verify_model,embed_text,verify_embeddings,embed_query_text,search_command

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_model_parser = subparsers.add_parser("verifymodel", help="Verify the model is loaded correctly")
    verify_model_parser.add_argument("model", type=str, nargs="?", default="all-MiniLM-L6-v2", help="Model to verify")
    embed_text_parser = subparsers.add_parser("embedtext", help="Embed a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")
    verify_embeddings_parser = subparsers.add_parser("verifyembeddings", help="Verify the embeddings are loaded correctly")
    embed_query_text_parser = subparsers.add_parser("embedquery", help="Embed a given query text")
    embed_query_text_parser.add_argument("query", type=str, help="Query text to embed")
    search_parser = subparsers.add_parser("search", help="Search for the most similar documents to a given query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    args = parser.parse_args()
    match args.command:
        case "verifyembeddings":
            verify_embeddings()
        case "verifymodel":
            verify_model(args.model)
        case "embedtext":
            print(embed_text(args.text))
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query)
        case _:
            print("Invalid command")


if __name__ == "__main__":
    main()