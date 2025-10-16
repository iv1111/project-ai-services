from pymilvus import connections, Collection
import math
import json

from llm_utils import generate_qa_pairs
from db_utils import MilvusVectorStore

def fetch_all_entries(collection_name, batch_size=1000):
    connections.connect(alias="default", host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()

    total = collection.num_entities
    num_batches = math.ceil(total / batch_size)

    # all_records = []
    # for i in range(num_batches):
    #     offset = i * batch_size
    #     records = collection.query(
    #         expr="",
    #         # output_fields=["page_content"],
    #         offset=offset,
    #         limit=batch_size
    #     )
    #     all_records.extend(records)

    MAX_LIMIT = 16384
    all_records = collection.query(
        expr="",
        output_fields=["page_content", "chunk_id"],
        offset=0,
        limit=MAX_LIMIT
    )

    print(f"✅ Retrieved {len(all_records)} records from Milvus DB {collection_name}.")
    return all_records


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
DB_NAME_PREFIX = 'BWI_Docs_V1'
emb_model = 'granite-embedding-278m-multilingual'
vlm_model = 'granite-vision-3.2-2b'
llm_model = 'granite-3.3-8b-instruct'
collection_name = MilvusVectorStore(
    MILVUS_HOST, MILVUS_PORT, DB_NAME_PREFIX, emb_model, vlm_model, llm_model
)._generate_collection_name()
records = fetch_all_entries(collection_name)
qa_pairs = generate_qa_pairs(
    records,
    "/wca4z-pvc-ckpt/HF_cache/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1",
    "https://akm-mixtral-8x7b-instruct-v0p1-vllm-code.apps.dmf.dipc.res.ibm.com/v1/completions"
)

with open(f"qa_pairs_{collection_name}.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"✅ QA pairs saved to qa_pairs_{collection_name}.json")
