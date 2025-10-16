import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from pymilvus import connections, list_collections, Collection

def convert_floats(obj):
    """
    Recursively converts numpy float32 to Python float (float64).
    """
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_floats(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_floats(value) for key, value in obj.items()}
    return obj

def retrieve_documents_with_filename_filter(collection, out_fields, filenames, limit=10000):
    """
    Retrieves documents from Milvus, partitioned by the `filename` field.
    The function uses a filter on `filename` to split the dataset into smaller queries.

    Parameters:
        collection (pymilvus.Collection): The Milvus collection object.
        filenames (list): List of filenames to filter by.
        limit (int): The limit for each query batch.

    Returns:
        list: A list of retrieved documents.
    """
    all_results = []
    
    for filename in tqdm(filenames):
        batch = collection.query(
            expr=f"filename == '{filename}'",  # Filter by filename
            limit=limit,
            output_fields=out_fields
        )
        
        # If no results are returned, skip the current batch
        if not batch:
            continue
        
        # Convert all numpy.float32 types to Python float
        batch = convert_floats(batch)
        
        # Append the batch to the results
        all_results.extend(batch)
        
        print(f"Retrieved {len(batch)} documents for filename: {filename}")
    
    return all_results

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# List collections
collections = list_collections()
if not collections:
    print("❌ No collections found.")
    exit(1)

print("\n📦 Available collections:")
for name in collections:
    print(f"  - {name}")

# Get user input
collection_name = input("\nEnter the collection name to export: ").strip()
if collection_name not in collections:
    print(f"❌ Collection '{collection_name}' does not exist.")
    exit(1)

# Load collection
collection = Collection(collection_name)
collection.load()

output_fields = [field.name for field in collection.schema.fields]

# Print available fields to confirm
print(f"\n📄 Fields in collection '{collection_name}': {output_fields}")

allowed_file_types = ['pdf', 'docx', 'html']
file_paths = []

# Collect all file paths
directory_path = input("\nEnter the directory path containing the documents: ").strip()
for f_type in allowed_file_types:
    file_paths.extend(glob(f'{directory_path}/**/*.{f_type}', recursive=True))

# Extract just the filenames (remove path)
filenames = [os.path.basename(file_path) for file_path in file_paths]

# Retrieve documents filtered by filenames
all_data = retrieve_documents_with_filename_filter(collection, output_fields, filenames)

# Save to JSON
output_file = f"{collection_name}_backup.json"
with open(output_file, "w") as f:
    json.dump(all_data, f)

print(f"\n✅ Export complete: {len(all_data)} records saved to '{output_file}'")



# # Paginated fetch
# batch_size = 1000
# offset = 0
# all_data = []

# print(f"\n🔄 Fetching data from '{collection_name}'...")
# while True:
#     batch = collection.query(
#         expr="",
#         output_fields=output_fields,
#         limit=batch_size,
#         offset=offset
#     )
#     if not batch:
#         break
#     all_data.extend(batch)
#     offset += batch_size
#     print(f"  ✅ Retrieved {len(batch)} records (Total: {len(all_data)})")
