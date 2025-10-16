"""

python3 prepare_data_for_rag_eval.py \
  --data_path "./qa_pairs_BWI_Docs_V1_4142bff4928f13415914eedf6627ca39.json" \
  --deployment cuda \
  --embedding_model granite-embedding-278m-multilingual \
  --vlm_model granite-vision-3.2-2b \
  --llm_db_model granite-3.3-8b-instruct \
  --llm_qa_model granite-3.3-8b-instruct \
  --reranker rerank-bge-reranker-large

"""


import json
import time
import numpy as np
from tqdm import tqdm
from pymilvus import connections

from misc_utils import get_model_endpoints
from reranker_utils import rerank_documents
from retrieval_utils import retrieve_documents
from db_utils import MilvusVectorStore
from llm_utils import query_vllm

# Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
TOP_K = 20
TOP_R = 5
DB_NAME_PREFIX = 'BWI_Docs_V1'

def initialize_vectorstore(emb_model_choice, vlm_model_choice, llm_model_choice):
    vectorstore = MilvusVectorStore(MILVUS_HOST, MILVUS_PORT, DB_NAME_PREFIX, emb_model_choice, vlm_model_choice, llm_model_choice)
    return vectorstore

def retrieve_n_answer(
    query, vectorstore, emb_model, emb_endpoint, emb_max_tokens, 
    deployment_type, reranker_model, reranker_endpoint, llm_model, llm_endpoint
):
    retrieved_documents, _ = retrieve_documents(
        query, emb_model, emb_endpoint, emb_max_tokens, vectorstore, TOP_K, deployment_type, mode="hybrid")
    
    reranked = rerank_documents(query, retrieved_documents, reranker_model, reranker_endpoint)

    reranked_docs = []
    reranked_doc_ids = []
    docs_for_rag = []
    for rank, (doc, score) in enumerate(reranked[:TOP_R], 1):
        reranked_docs.append(doc.get("page_content"))
        reranked_doc_ids.append(int(doc.get("chunk_id")))
        docs_for_rag.append(doc)

    stop_words = ['### Response:', 'Answer:', '### Instruction:', 'Input:']

    rag_answer, rag_generation_time = query_vllm(
        query, docs_for_rag, llm_endpoint, llm_model, language="English", stop_words=stop_words, rag=True
    )

    rag_answer = rag_answer.get('choices', [{}])[0].get('text')

    return rag_answer, reranked_docs, reranked_doc_ids

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="JSON file with 'question', 'chunk_id', ...")
    parser.add_argument("--deployment", type=str, default="cpu", choices=["cpu", "cuda", "spyre"])
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--vlm_model", required=True)
    parser.add_argument("--llm_db_model", required=True)
    parser.add_argument("--llm_qa_model", required=True)
    parser.add_argument("--reranker", required=True)
    args = parser.parse_args()

    # Connect to Milvus
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    deployment_type = args.deployment.strip().lower()

    # Load endpoints
    emb_model_dict, _, llm_model_dict, reranker_model_dict = get_model_endpoints(deployment_type)
    emb_model = emb_model_dict[args.embedding_model]['emb_model']
    emb_endpoint = emb_model_dict[args.embedding_model]['emb_endpoint']
    emb_max_tokens = emb_model_dict[args.embedding_model]['max_tokens']
    reranker_model = reranker_model_dict[args.reranker]['reranker_model']
    reranker_endpoint = reranker_model_dict[args.reranker]['reranker_endpoint']
    llm_model = llm_model_dict[args.llm_qa_model]['llm_model']
    llm_endpoint = llm_model_dict[args.llm_qa_model]['llm_endpoint']

    # Init Vectorstore
    vectorstore = initialize_vectorstore(
        args.embedding_model, args.vlm_model, args.llm_db_model
    )

    # Load JSON test data
    with open(args.data_path, "r") as f:
        examples = json.load(f)

    np.random.seed(42)
    examples = np.random.choice(examples, size=500, replace=False)
    gt_data = []
    pred_data = []
    for ex_id, ex in tqdm(enumerate(examples, 1), total=len(examples)):
        question = ex["question"]
        reference_answer = ex["answer"]
        reference_context = ex["context"]
        reference_context_id = ex["chunk_id"]

        if question:
            start = time.time()
            gt_data.append(
                {
                    "question": question,
                    "question_id": ex_id,
                    "reference_answers": [reference_answer],
                    "reference_contexts": [reference_context],
                    "reference_context_ids": [reference_context_id],
                    "is_answerable_label": True,
                }
            )
            predicted_answer, retrieved_contexts, retrieved_context_ids = retrieve_n_answer(
                question, vectorstore, emb_model, emb_endpoint, emb_max_tokens, 
                deployment_type, reranker_model, reranker_endpoint, llm_model, llm_endpoint
            )
            pred_data.append(
                {
                    "answer": predicted_answer,
                    "contexts": retrieved_contexts,
                    "context_ids": retrieved_context_ids,
                    "is_answerable": True,
                }
            )
            # print(f"[{ex_id}/{len(examples)}] Time: {time.time() - start:.2f}s")
        else:
            continue
    
    collection_name = vectorstore._generate_collection_name()

    with open(f"gt_data_{collection_name}.json", "w") as f:
        json.dump(gt_data, f, indent=2)
    with open(f"pred_data_{collection_name}.json", "w") as f:
        json.dump(pred_data, f, indent=2)

if __name__ == "__main__":
    main()
