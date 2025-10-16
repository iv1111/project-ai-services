import os
import time
import shutil
import gradio as gr
from glob import glob
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from misc_utils import get_txt_img_tab_filenames, get_model_endpoints
from db_utils import MilvusVectorStore
from doc_utils import extract_document_data, hierarchical_chunk_with_token_split, create_chunk_documents



# ---- Config for OpenAI API, vLLM API, and Milvus Connection ----
DEPLOYMENT_TYPE = 'cuda'
SELECTED_EMB = 'emb-me5-large'
SELECTED_VLM = 'pixtral-12b-2409'
SELECTED_LLM = 'granite-3.3-8b-instruct'
SELECTED_TRANSLATOR = 'phi-4'
emb_model_dict, vlm_model_dict, llm_model_dict, _, translation_model_dict = get_model_endpoints(DEPLOYMENT_TYPE)


def create_vector_db_from_directory(
        directory_path, selected_lang, reset_db, include_meta_info_in_main_text, 
        log_messages=''
    ):
    # Initialize/reset the database before processing any files
    db_prefix = f'{selected_lang}'
    vector_store = MilvusVectorStore(db_prefix=db_prefix, emb_name=SELECTED_EMB, vlm_name=SELECTED_VLM, llm_name=SELECTED_LLM)
    collection_name = vector_store._generate_collection_name()
    if reset_db:
        vector_store.reset_collection()
        if os.path.isdir(f'images_{collection_name}'):
            shutil.rmtree(f'images_{collection_name}')
        log_messages += f"Resetting Vector DB: {collection_name}\n"
        yield gr.update(value=log_messages)
    
    # Process each document in the directory
    allowed_file_types = ['pdf', 'docx', 'html', 'odt']
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
        file_paths, out_path, llm_model_dict[SELECTED_LLM]['llm_model'], llm_model_dict[SELECTED_LLM]['llm_endpoint'], 
        vlm_model_dict[SELECTED_VLM]['vlm_model'], vlm_model_dict[SELECTED_VLM]['vlm_endpoint'], vlm_model_dict[SELECTED_VLM]['hosting_type'])

    original_filenames, input_txt_files, input_img_files, input_tab_files = get_txt_img_tab_filenames(file_paths, out_path)
    output_chunk_files = [f.replace('_clean_text.json', '_clean_chunk.json') for f in input_txt_files]
    hierarchical_chunk_with_token_split(
        input_txt_files, output_chunk_files, 
        max_tokens=emb_model_dict[SELECTED_EMB]['max_tokens'] - 100 if include_meta_info_in_main_text else emb_model_dict[SELECTED_EMB]['max_tokens']
    )
    combined_filtered_chunks = []
    for in_txt_f, in_img_f, in_tab_f, orig_fn, out_txt_f in tqdm(zip(
        input_txt_files, input_img_files, input_tab_files, original_filenames, output_chunk_files
    ), total=len(input_txt_files), desc='Creating Chunks'):
        # Combine all chunks (text, image summaries, table summaries)
        filtered_chunks, stats = create_chunk_documents(
            out_txt_f, in_img_f, in_tab_f, orig_fn, include_meta_info_in_main_text, collection_name, 
            translation_model_dict[SELECTED_TRANSLATOR]['translator_model'], translation_model_dict[SELECTED_TRANSLATOR]['translator_endpoint']
        )
        combined_filtered_chunks.extend(filtered_chunks)
        log_messages += f"{orig_fn}: {stats}\n"
        yield gr.update(value=log_messages)
    
    # Insert data into Milvus
    vector_store.insert_chunks(
        emb_model=emb_model_dict[SELECTED_EMB]['emb_model'],
        emb_endpoint=emb_model_dict[SELECTED_EMB]['emb_endpoint'],
        max_tokens=emb_model_dict[SELECTED_EMB]['max_tokens'],
        chunks=combined_filtered_chunks,
        deployment_type=DEPLOYMENT_TYPE
    )
    
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
            gr.Dropdown(label="Select Data Domain", choices=['docling_agri_docs_v1_hi_en']),
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
