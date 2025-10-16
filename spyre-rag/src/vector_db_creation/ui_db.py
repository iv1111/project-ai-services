import os
import time
import shutil
import gradio as gr
from glob import glob
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from misc_utils import get_txt_img_tab_filenames, get_model_endpoints
from db_utils import generate_collection_name, reset_db_in_milvus, insert_data_to_milvus
from doc_utils import extract_document_data, hierarchical_chunk_with_token_split, create_chunk_documents



# ---- Config for OpenAI API, vLLM API, and Milvus Connection ----
DEPLOYMENT_TYPE = 'cpu'
selected_embedding_model_name = 'granite-embedding-278m-multilingual'
selected_vlm = 'granite-vision-3.2-2b'
selected_llm = 'granite-3.3-8b-instruct'
DB_NAME_PREFIX = 'BWI_Docs_V1'
emb_model_dict, vlm_model_dict, llm_model_dict, _ = get_model_endpoints(DEPLOYMENT_TYPE)


def create_vector_db_from_directory(directory_path, emb_name, vlm_name, llm_name, reset_db, include_meta_info_in_main_text, log_messages=''):
    # Initialize/reset the database before processing any files
    collection_name = generate_collection_name(emb_name, vlm_name, llm_name, DB_NAME_PREFIX)
    if reset_db:
        reset_db_in_milvus(collection_name)
        if os.path.isdir(f'images_{collection_name}'):
            shutil.rmtree(f'images_{collection_name}')
        log_messages += f"Resetting Vector DB: {collection_name}\n"
        yield gr.update(value=log_messages)
    
    # Process each document in the directory
    allowed_file_types = ['pdf', 'docx', 'html']
    file_paths = []
    for f_type in allowed_file_types:
        file_paths.extend(glob(f'{directory_path}/**/*.{f_type}', recursive=True))
    files_being_processed = '\n'.join(f for f in file_paths)
    log_messages += f"\nProcessing the following files:\n{files_being_processed}\n"
    yield gr.update(value=log_messages)

    out_path = f'{collection_name}_cache'
    os.makedirs(out_path, exist_ok=True)

    start_time = time.time()
    extract_document_data(
        file_paths, out_path, llm_model_dict[llm_name]['llm_model'], llm_model_dict[llm_name]['llm_endpoint'], 
        vlm_model_dict[vlm_name]['vlm_model'], vlm_model_dict[vlm_name]['vlm_endpoint'], vlm_model_dict[vlm_name]['hosting_type'])

    original_filenames, input_txt_files, input_img_files, input_tab_files = get_txt_img_tab_filenames(file_paths, out_path)
    output_chunk_files = [f.replace('_clean_text.json', '_clean_chunk.json') for f in input_txt_files]
    hierarchical_chunk_with_token_split(
        input_txt_files, output_chunk_files, 
        max_tokens=emb_model_dict[emb_name]['max_tokens'] - 50 if include_meta_info_in_main_text else emb_model_dict[emb_name]['max_tokens']
    )
    combined_filtered_chunks = []
    for in_txt_f, in_img_f, in_tab_f, orig_fn, out_txt_f in zip(input_txt_files, input_img_files, input_tab_files, original_filenames, output_chunk_files):
        # Combine all chunks (text, image summaries, table summaries)
        filtered_chunks, stats = create_chunk_documents(out_txt_f, in_img_f, in_tab_f, orig_fn, include_meta_info_in_main_text, collection_name)
        combined_filtered_chunks.extend(filtered_chunks)
        log_messages += f"{orig_fn}: {stats}\n"
        yield gr.update(value=log_messages)
    
    # Insert data into Milvus
    insert_data_to_milvus(
        emb_name, vlm_name, llm_name, DB_NAME_PREFIX, 
        emb_model_dict[emb_name]['emb_model'],
        emb_model_dict[emb_name]['emb_endpoint'],
        emb_model_dict[emb_name]['max_tokens'],
        combined_filtered_chunks)
    log_messages += f"Inserted {len(combined_filtered_chunks)} chunks to the vector DB: {collection_name}\n"
    yield gr.update(value=log_messages)

    # Log time taken for the file
    end_time = time.time()  # End the timer for the current file
    file_processing_time = end_time - start_time
    log_messages += f"Time taken to ingest data in vector DB is: {file_processing_time:.2f} seconds\n\n"
    yield gr.update(value=log_messages)

    log_messages += f"\nVector DB ({collection_name}) creation completed successfully!\n"
    yield gr.update(value=log_messages)


# ---- Gradio Interface ----
def run_gradio_app(server_port):
    demo = gr.Interface(
        fn=create_vector_db_from_directory,
        inputs=[
            gr.Textbox(label="Directory Path", placeholder="Enter the path to the documents folder"),
            gr.Dropdown(label="Select Embedding Model", choices=list(emb_model_dict.keys()), value=selected_embedding_model_name),
            gr.Dropdown(label="Select Vision Model (for Image Summarization)", choices=list(vlm_model_dict.keys()), value=selected_vlm),
            gr.Dropdown(label="Select LLM Model (for Table Summarization and Data Cleaning)", choices=list(llm_model_dict.keys()), value=selected_llm),
            gr.Checkbox(label="Reset DB (Delete existing data)", value=False),
            gr.Checkbox(label="Include meta-data in text chunk", value=False),
        ],
        outputs=gr.Textbox(value="", label="Process Progress", interactive=False),
        title="🛠️ Milvus Vector DB Ingestion Tool",
        allow_flagging="never"
    )

    demo.launch(server_port=server_port, server_name="0.0.0.0")

# ---- Run Gradio App ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for DB Ingestion Service Deployment.")
    parser.add_argument('-p','--port_no', help='Port number for the service', type=int, default=20000)
    args = parser.parse_args()
    run_gradio_app(server_port=args.port_no)
