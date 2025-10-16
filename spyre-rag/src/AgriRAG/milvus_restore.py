import os
import json
import numpy as np
from tqdm import tqdm
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


def restore_data_to_milvus(collection_name, backup_file, dim=768, host='localhost', port='19530', batch_size=1000):
    """
    Restore backed-up data into a Milvus collection.

    Parameters:
        collection_name (str): The name of the Milvus collection to insert data into.
        backup_file (str): The path to the backed-up JSON file.
        host (str): The Milvus server host.
        port (str): The Milvus server port.
    """

    # Define the schema for the collection
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=32768, enable_analyzer=True),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32768),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=8),
    ]
    schema = CollectionSchema(fields=fields, description="RAG chunk storage (dense only)")

    # Connect to Milvus
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")
    
    # Load or create collection
    if collection_name not in utility.list_collections():
        print(f"Collection '{collection_name}' does not exist, creating it.")
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(collection_name)
    
    # Check if the index exists
    if not collection.has_index():
        print(f"No index found for collection '{collection_name}'. Creating index...")
        
        # Create the index for the 'embedding' field
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )
        print(f"Index created for 'embedding' field in collection '{collection_name}'.")
    
    # Load the collection to ensure it is ready for queries
    collection.load()
    
    # Load the backup data (assuming it's a list of documents)
    with open(backup_file, 'r') as f:
        backup_data = json.load(f)
    
    # Prepare the data for insertion
    vectors = []
    page_contents = []
    filenames = []
    types = []
    sources = []
    chunk_ids = []
    language = []
    
    for doc in backup_data:
        vectors.append(doc['embedding'])
        page_contents.append(doc.get('page_content', ''))
        filenames.append(doc.get('filename', ''))
        types.append(doc.get('type', ''))
        sources.append(doc.get('source', ''))
        chunk_ids.append(doc.get('chunk_id', None))
        language.append(doc.get('language', None))

    
    # Prepare the data to insert into Milvus (ensure vectors are in numpy array format)
    vector_data = np.array(vectors)
    
    # Insert the data back into the Milvus collection
    insert_data = [
        chunk_ids,
        vector_data.tolist(),
        page_contents,
        filenames,
        types,
        sources,
        language
    ]
    
    # Insert in smaller batches
    for i in tqdm(range(0, len(backup_data), batch_size), desc='Restoring data in the Milvus DB'):
        batch = [data[i:i + batch_size] for data in insert_data]
        collection.insert(batch)

    print(f"Data injection complete!")
    collection.release()
    print(f"Injected {len(backup_data)} documents into collection '{collection_name}'")

    # Release resources
    collection.release()
    print(f"Data injection complete!")

# Example usage:
backup_file = "./docling_agri_docs_v1_hi_en_52c77c4b6668e08e35fce7a0187fe70a_backup.json"
collection_name = "docling_agri_docs_v1_hi_en_52c77c4b6668e08e35fce7a0187fe70a"
restore_data_to_milvus(collection_name, backup_file)

backup_file = "./docling_agri_docs_v1_hi_en_b3e0264d3bd131966928e10ca1b13ae4_backup.json"
collection_name = "docling_agri_docs_v1_hi_en_b3e0264d3bd131966928e10ca1b13ae4"
restore_data_to_milvus(collection_name, backup_file, dim=768)

backup_file = "./docling_agri_docs_v1_hi_en_fb9e10c1de72c10a83925420ad9caf90_backup.json"
collection_name = "docling_agri_docs_v1_hi_en_fb9e10c1de72c10a83925420ad9caf90"
restore_data_to_milvus(collection_name, backup_file, dim=1024)