import asyncio
import hashlib
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import Collection, connections, list_collections

from emb_utils import FastAPIEmbeddingFunction

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# ---- Function to Generate Unique Collection Name ----
def generate_collection_name(emb_name, vlm_name, llm_name, db_name_prefix):
    """
    Generates a unique collection name based on the selected models.
    This combines the names of the embedding models and generation model, and hashes them to ensure uniqueness.
    """
    # Combine the model names to create a base string
    base_name = f"{emb_name}_{vlm_name}_{llm_name}"
    
    # Hash the base name to create a unique identifier (this prevents long or repetitive names)
    collection_name = hashlib.md5(base_name.encode()).hexdigest()
    
    return f'{db_name_prefix}_{collection_name}'


# ---- Function to Set Up Milvus Collection ----
def milvus_collection_setup(collection_name, embeddings):
    """
    Set up a Milvus collection using LangChain's Milvus vector store and BM25 built-in function.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    embeddings = embeddings
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        consistency_level="Strong"
    )

    return vectorstore

# ---- Example of Usage ----
def setup_milvus_for_selected_models(emb_name, vlm_name, llm_name, db_name_prefix, emb_model, emb_endpoint, max_tokens):
    # Generate the collection name dynamically based on the selected models
    collection_name = generate_collection_name(emb_name, vlm_name, llm_name, db_name_prefix)
    
    # Set up the collection with the required fields
    embeddings = FastAPIEmbeddingFunction(emb_model, emb_endpoint, max_tokens)
    collection = milvus_collection_setup(collection_name, embeddings)
    return collection

# ---- Function to Reset Milvus DB (Delete existing collection) ----
def reset_db_in_milvus(collection_name):
    # Connect to Milvus
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Check if the collection exists
    if collection_name in list_collections():
        # If collection exists, delete it
        collection = Collection(collection_name)
        collection.drop()  # Drops the collection
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist. Nothing to delete.")


def insert_data_to_milvus(emb_name, vlm_name, llm_name, db_name_prefix, emb_model, emb_endpoint, max_tokens, all_chunks):
    """
    Insert text chunks and their embeddings into Milvus using LangChain vector store with BM25.
    """
    # Step 1: Create or connect to Milvus collection
    vector_store = setup_milvus_for_selected_models(emb_name, vlm_name, llm_name, db_name_prefix, emb_model, emb_endpoint, max_tokens)

    # Step 3: Insert into Milvus (LangChain manages the process)
    vector_store.add_documents(all_chunks)
    print(f"Inserted {len(all_chunks)} chunks into Milvus.")




# def initialize_vectorstore(emb_model_choice, vlm_model_choice, llm_model_choice, db_name_prefix, emb_model, emb_endpoint, max_tokens):
#     global vectorstore
#     import asyncio
#     try:
#         asyncio.get_running_loop()
#     except RuntimeError:
#         asyncio.set_event_loop(asyncio.new_event_loop())

#     embeddings = FastAPIEmbeddingFunction(emb_model, emb_endpoint, max_tokens)
#     collection_name = generate_collection_name(emb_model_choice, vlm_model_choice, llm_model_choice, db_name_prefix)
#     try:
#         vectorstore = Milvus(
#             embedding_function=embeddings,
#             collection_name=collection_name,
#             builtin_function=BM25BuiltInFunction(),
#             vector_field=["dense", "sparse"],
#             connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
#             consistency_level="Strong"
#         )
#         return vectorstore
#     except Exception as e:
#         print(f"Error initializing vectorstore: {e}")
