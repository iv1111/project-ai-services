from flask import Flask, request, jsonify
import time
from pymilvus import connections
from langchain_milvus import Milvus

from misc_utils import get_model_endpoints
from reranker_utils import rerank_documents
from retrieval_utils import retrieve_documents
from emb_utils import FastAPIEmbeddingFunction
from db_utils import MilvusVectorStore, VectorStoreManager

# === Static config (same as your CLI args) ===
MILVUS_HOST = "localhost" # "localhost"
MILVUS_PORT = "19530"
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

TOP_K = 20
TOP_R = 3
DB_NAME_PREFIX = 'BWI_Docs_V1'

# Fixed deployment and models (same as CLI)
DEPLOYMENT = "cpu"
EMBEDDING_MODEL = "granite-embedding-278m-multilingual"
VLM_MODEL = "granite-vision-3.2-2b"
LLM_DB_MODEL = "granite-3.3-8b-instruct"
RERANKER = "rerank-bge-reranker-large"

# Initialize global variables
model_endpoints = {}
vectorstore = None
reranker_model = None
reranker_endpoint = None


vector_store_manager = VectorStoreManager()



def initialize_vectorstore(emb_model_choice, vlm_model_choice, llm_model_choice, db_name_prefix,
                           emb_model, emb_endpoint, max_tokens):

    embeddings = FastAPIEmbeddingFunction(emb_model, emb_endpoint, max_tokens)
    #collection_name = generate_collection_name(emb_model_choice, vlm_model_choice, llm_model_choice, db_prefix)
    print("Here -----")
    print(db_name_prefix)
    print(emb_model_choice)
    print(vlm_model_choice)
    print(llm_model_choice)

    current_config = {
        "emb": emb_model_choice,
        "vlm": vlm_model_choice,
        "llm": llm_model_choice,
        "db_prefix": db_name_prefix,
    }

    config_changed = current_config != vector_store_manager.last_config



    if vector_store_manager.vectorstore is None or config_changed:
        print("🔄 Reinitializing vectorstore due to config change...")

        vectorstore = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            db_prefix=db_name_prefix,
            emb_name=emb_model_choice,
            vlm_name=vlm_model_choice,
            llm_name=llm_model_choice
        )

        try:
            # Access the underlying Milvus collection (replace with actual attribute)
            collection = vectorstore.collection

            # List indexes on this collection
            indexes = collection.indexes

            sparse_index_found = False
            for idx in indexes:
                print(f"Found index: {idx.name} type: {idx.params.get('index_type')}")
                if 'SPARSE' in idx.params.get('index_type', '').upper():
                    sparse_index_found = True
                    break

            if not sparse_index_found:
                print("⚠️ Sparse index missing for hybrid search.")
                # Optionally raise an exception or handle this case
                # raise RuntimeError("Sparse index missing for hybrid search.")

        except Exception as e:
            print(f"Failed to check sparse index: {e}")


        vector_store_manager.vectorstore = vectorstore
        vector_store_manager.last_config = current_config
        return vectorstore

    print("✅ Reusing existing vectorstore.")
    return vector_store_manager.vectorstore


def create_app():
    global model_endpoints, vectorstore, reranker_model, reranker_endpoint

    app = Flask(__name__)

    print("🔌 Connecting to Milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    print("🔧 Loading model configurations...")
    model_endpoints = get_model_endpoints(DEPLOYMENT)
    print("✅ Model endpoints loaded.")

    emb_model_dict, _, _, reranker_model_dict = model_endpoints

    emb_model = emb_model_dict[EMBEDDING_MODEL]['emb_model']
    emb_endpoint = emb_model_dict[EMBEDDING_MODEL]['emb_endpoint']
    emb_max_tokens = emb_model_dict[EMBEDDING_MODEL]['max_tokens']

    vectorstore = initialize_vectorstore(
        EMBEDDING_MODEL, VLM_MODEL, LLM_DB_MODEL, DB_NAME_PREFIX,
        emb_model, emb_endpoint, emb_max_tokens
    )

    reranker_model = reranker_model_dict[RERANKER]['reranker_model']
    reranker_endpoint = reranker_model_dict[RERANKER]['reranker_endpoint']

    @app.route("/retrieve", methods=["POST"])
    def retrieve_documents_api():
        try:
            print("here before")
            data = request.json or {}
            query = data.get("query")
            if not query:
                return jsonify({"error": "Missing required field: query"}), 400
            
            print("now here")
            retrieval_start = time.time()
            #print("giving these arguments")
            #print(query)
            #print(emb_model)
            #print(emb_endpoint)
            #print(emb_max_tokens)
            retrieved_documents, scores = retrieve_documents(query, emb_model, emb_endpoint, emb_max_tokens, vectorstore, TOP_K, deployment_type="cpu")
            #print("here")
            #print(retrieved_documents)
            print("before renaked")
            print(reranker_endpoint)
            print(reranker_model)
            reranked = rerank_documents(query, retrieved_documents, reranker_model, reranker_endpoint)
            retrieval_end = time.time()
            print(reranked)
            #retrieval_end = time.time()
            print("HERE!")
            print("========")
            print(reranked)
            print("========")
            #print(ranked_documents)
            #print(ranked_scores)

            results = []
            for i, (doc_dict, score) in enumerate(reranked, 1):
                if (i > TOP_R):
                    break
                results.append({
                    "metadata": {
                    # Add anything else you'd like included here
                    "filename": doc_dict.get("filename"),
                },
                "page_content": doc_dict.get("page_content", ""),
                "type": doc_dict.get("type", ""),
                "source": doc_dict.get("source", "") #Is going to be table, picture or section from the text
            })
            print("results")
            print("==================")
            print(results)

            return jsonify({
                "query": query,
                "results": results,
                "retrieval_time_seconds": round(retrieval_end - retrieval_start, 3)
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8084, debug=True)

