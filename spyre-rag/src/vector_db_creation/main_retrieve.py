"""

python3 main_retrieve.py \
  --query "Can I do LPM between data centers?" \
  --deployment cuda \
  --embedding_model granite-embedding-278m-multilingual \
  --vlm_model granite-vision-3.2-2b \
  --llm_db_model granite-3.1-8b-instruct \
  --reranker rerank-bge-reranker-large

"""


import time
import argparse
from pymilvus import connections
from langchain_milvus import BM25BuiltInFunction, Milvus

from misc_utils import get_model_endpoints
from reranker_utils import rerank_documents
from retrieval_utils import retrieve_documents
from emb_utils import FastAPIEmbeddingFunction
from db_utils import generate_collection_name

# Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OpenAI_API_KEY = "EMPTY"
TOP_K = 20
TOP_R = 5
DB_NAME_PREFIX = 'BWI'


def initialize_vectorstore(emb_model_choice, vlm_model_choice, llm_model_choice, db_prefix,
                           emb_model, emb_endpoint, max_tokens):

    embeddings = FastAPIEmbeddingFunction(emb_model, emb_endpoint, max_tokens)
    collection_name = generate_collection_name(emb_model_choice, vlm_model_choice, llm_model_choice, db_prefix)

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        consistency_level="Strong"
    )

    print(f"✅ Connected to collection: {collection_name}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="RAG Retriever CLI")
    parser.add_argument("--query", type=str, required=True, help="Your input question/query")
    parser.add_argument("--deployment", type=str, default="cpu", choices=["cpu", "cuda", "spyre"])
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--vlm_model", required=True)
    parser.add_argument("--llm_db_model", required=True)
    parser.add_argument("--reranker", required=True)
    args = parser.parse_args()

    # Connect to Milvus
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Load model configs
    emb_model_dict, _, _, reranker_model_dict = get_model_endpoints(args.deployment)

    # Init vectorstore
    emb_model = emb_model_dict[args.embedding_model]['emb_model']
    emb_endpoint = emb_model_dict[args.embedding_model]['emb_endpoint']
    emb_max_tokens = emb_model_dict[args.embedding_model]['max_tokens']

    vectorstore = initialize_vectorstore(
        args.embedding_model, args.vlm_model, args.llm_db_model,
        DB_NAME_PREFIX, emb_model, emb_endpoint, emb_max_tokens
    )

    # Get retrieval parameters
    reranker_model = reranker_model_dict[args.reranker]['reranker_model']
    reranker_endpoint = reranker_model_dict[args.reranker]['reranker_endpoint']

    # Retrieve docs
    retrieval_start = time.time()
    retrieved_documents, scores = retrieve_documents(args.query, vectorstore, TOP_K)
    reranked = rerank_documents(args.query, retrieved_documents, reranker_model, reranker_endpoint)
    ranked_documents = []
    ranked_scores = []
    for i, (doc, score) in enumerate(reranked, 1):
        ranked_documents.append(doc)
        ranked_scores.append(score)
        if i == TOP_R:
            break
    retrieval_end = time.time()

    print("\n🔍 Top Retrieved Documents:\n")
    for doc in ranked_documents:
        print(doc)
        print('-' * 100)
    print(f"Time Required for Retrieval and Reranking: {retrieval_end - retrieval_start}")


if __name__ == "__main__":
    main()
