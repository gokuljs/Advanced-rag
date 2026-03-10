import argparse
from lib.semantic_search import verify_model,embed_text,verify_embeddings,embed_query_text,search_command,chunk_text,semantic_chunk_command,embed_chunks

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
    chunk_text_parser = subparsers.add_parser("chunktext", help="Chunk a given text into fixed size chunks")
    chunk_text_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_text_parser.add_argument("chunk_size", type=int, help="Chunk size", nargs="?", default=200)
    chunk_text_parser.add_argument("overlap", type=int, help="Overlap", nargs="?", default=20)
    semantic_chunk_parser = subparsers.add_parser("semanticchunk", help="Chunk text on sentence boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Sentence overlap between chunks")
    embed_chunks_parser = subparsers.add_parser("embedchunks", help="Embed chunks of a given text")
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
        case "chunktext":
            chunk_text(args.text,args.overlap, args.chunk_size,)
        case "semanticchunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embedchunks":
            embed_chunks()
        case _:
            print("Invalid command")


if __name__ == "__main__":
    main()