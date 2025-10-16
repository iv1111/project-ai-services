import gradio as gr
import argparse
from pymilvus import connections
from misc_utils import get_model_endpoints
from retrieval_utils import search_and_answer_dual, search_and_answer
from emb_utils import FastAPIEmbeddingFunction
from db_utils import MilvusVectorStore, VectorStoreManager

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

OpenAI_API_KEY = "EMPTY"
TOP_K = 20
TOP_R = 5

# Default DB prefix pattern
DB_NAME_PREFIX = 'docling_agri_docs_v1_hi_en'
vectorstore = None
# Keep track of the last used config for vectorstore
DEPLOYMENT_TYPE = 'cuda'
SELECTED_EMB = 'emb-me5-large'
SELECTED_VLM = 'pixtral-12b-2409'
SELECTED_LLM = 'granite-3.3-8b-instruct'

# Globals to be set dynamically
emb_model_dict = {}
vlm_model_dict = {}
llm_model_dict = {}
reranker_model_dict = {}

# Keep track of the last used config for vectorstore
vector_store_manager = VectorStoreManager()

def initialize_models(deployment_type):
    global emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict
    emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict, _ = get_model_endpoints(deployment_type)

    return (
        gr.update(choices=list(llm_model_dict.keys()), value=list(llm_model_dict.keys())[-1]),
        gr.update(choices=list(reranker_model_dict.keys()), value=list(reranker_model_dict.keys())[-1]),
    )

def initialize_vectorstore_if_needed(db_name_prefix):
    current_config = {
        "emb": SELECTED_EMB,
        "vlm": SELECTED_VLM,
        "llm": SELECTED_LLM,
        "db_prefix": db_name_prefix,
    }

    config_changed = current_config != vector_store_manager.last_config

    if vector_store_manager.vectorstore is None or config_changed:
        print("🔄 Reinitializing vectorstore due to config change...")
        vectorstore = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            db_prefix=db_name_prefix,
            emb_name=SELECTED_EMB,
            vlm_name=SELECTED_VLM,
            llm_name=SELECTED_LLM
        )
        vector_store_manager.vectorstore = vectorstore
        vector_store_manager.last_config = current_config
        return vectorstore

    print("✅ Reusing existing vectorstore.")
    return vector_store_manager.vectorstore

def detect_en_hi(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text)

    if total_chars == 0:
        return "unknown"

    if hindi_chars / total_chars > 0.3:
        return "hi"
    else:
        return "en"

def wrapper_search_and_answer_dual(
    question, llm_for_qa, reranker, deployment_type, num_chunks_dense, num_chunks_sparse, num_chunks_post_rrf, num_docs_reranker, use_in_context, use_reranker, max_tokens
):
    try:
        emb_model = emb_model_dict[SELECTED_EMB]['emb_model']
        emb_endpoint = emb_model_dict[SELECTED_EMB]['emb_endpoint']
        emb_max_tokens = emb_model_dict[SELECTED_EMB]['max_tokens']
        llm_model = llm_model_dict[llm_for_qa]['llm_model']
        llm_endpoint = llm_model_dict[llm_for_qa]['llm_endpoint']
        reranker_model = reranker_model_dict[reranker]['reranker_model']
        reranker_endpoint = reranker_model_dict[reranker]['reranker_endpoint']

        vectorstore = initialize_vectorstore_if_needed(DB_NAME_PREFIX)

        stop_words = ""

        (rag_ans, no_rag_ans, docs) = search_and_answer_dual(
            question,
            llm_endpoint,
            llm_model,
            emb_model, emb_endpoint, emb_max_tokens,
            reranker_model,
            reranker_endpoint,
            num_chunks_dense,
            num_chunks_sparse,
            num_chunks_post_rrf,
            num_docs_reranker,
            use_in_context,
            use_reranker,
            max_tokens,
            stop_words=stop_words,
            language=detect_en_hi(question),
            vectorstore=vectorstore,
            deployment_type=deployment_type,
            stream=False
        )

        return rag_ans, no_rag_ans, docs

    except Exception as e:
        error_html = f"<pre style='color:red;'>Error: {repr(e)}</pre>"
        return error_html, "", ""

def wrapper_search_and_answer(
    question, llm_for_qa, reranker, deployment_type, num_chunks_post_rrf, num_docs_reranker, use_in_context, use_reranker, max_tokens
):
    try:
        emb_model = emb_model_dict[SELECTED_EMB]['emb_model']
        emb_endpoint = emb_model_dict[SELECTED_EMB]['emb_endpoint']
        emb_max_tokens = emb_model_dict[SELECTED_EMB]['max_tokens']
        llm_model = llm_model_dict[llm_for_qa]['llm_model']
        llm_endpoint = llm_model_dict[llm_for_qa]['llm_chat_endpoint']
        reranker_model = reranker_model_dict[reranker]['reranker_model']
        reranker_endpoint = reranker_model_dict[reranker]['reranker_endpoint']

        vectorstore = initialize_vectorstore_if_needed(DB_NAME_PREFIX)

        stop_words = ""

        (rag_ans, docs) = search_and_answer(
            question,
            llm_endpoint,
            llm_model,
            emb_model, emb_endpoint, emb_max_tokens,
            reranker_model,
            reranker_endpoint,
            num_chunks_post_rrf,
            num_docs_reranker,
            use_in_context,
            use_reranker,
            max_tokens,
            stop_words=stop_words,
            language=detect_en_hi(question),
            vectorstore=vectorstore,
            deployment_type=deployment_type,
            stream=False
        )

        return rag_ans, docs

    except Exception as e:
        error_html = f"<pre style='color:red;'>Error: {repr(e)}</pre>"
        return error_html, ""

def run_gradio_app(server_port):
    deployment_type = gr.Dropdown(
        label="Deployment Type",
        choices=["cuda"],
        value="cuda"
    )

    llm_for_qa = gr.Dropdown(label="LLM for Generation")
    reranker = gr.Dropdown(label="Reranker Model")
    question = gr.Textbox(label="Enter your Question", placeholder="Type your question here...", lines=2)

    num_chunks_post_rrf = gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Number of Chunks to Retrieve")
    num_docs_reranker = gr.Slider(minimum=1, maximum=50, step=1, value=3, label="Number of Context to Feed")
    use_in_context = gr.Checkbox(label="Use In-Context Example", value=True)
    use_reranker = gr.Checkbox(label="Use Reranker", value=True)
    max_tokens = gr.Number(value=512, label="Max Output Tokens from Generator")
    
    init_button = gr.Button("Commit Deployment Type")
    with gr.Blocks(title="🤖 AgriRAG Demo") as demo:
        gr.Markdown("# 🤖 AgriRAG Demo (for BharatGen)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Configuration Panel")
                deployment_type.render()
                init_button.render()

                llm_for_qa.render()
                reranker.render()
                question.render()

                num_chunks_post_rrf.render()
                num_docs_reranker.render()
                use_in_context.render()
                use_reranker.render()
                max_tokens.render()

                with gr.Row():
                    run_button = gr.Button("Search and Answer")
                    clear_button = gr.Button("Clear")

            demo.css = """
                #rag-output, #no-rag-output {
                    height: 800px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                    background-color: #f8f8f8;
                    font-family: monospace;
                    white-space: pre-wrap;
                }

                #docs-output {
                    height: 800px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                    background-color: #f8f8f8;
                    font-family: monospace;
                    white-space: pre-wrap;
                }
                """
            
            with gr.Column():
                gr.Markdown("### Answer")
                rag_output = gr.HTML(label="RAG Answer", value="<h3>RAG Answer</h3>", elem_id="rag-output")

            with gr.Column():
                gr.Markdown("### Retrieved Documents")
                docs_out = gr.HTML(label="Top Documents", value="<h3>Top Documents</h3>", elem_id="docs-output")

        init_button.click(
            fn=initialize_models,
            inputs=[deployment_type],
            outputs=[
                llm_for_qa,
                reranker,
            ]
        )

        run_button.click(
            wrapper_search_and_answer,
            inputs=[
                question, llm_for_qa, reranker, deployment_type, num_chunks_post_rrf,
                num_docs_reranker, use_in_context, use_reranker, max_tokens
            ],
            outputs=[rag_output, docs_out]
        )

        clear_button.click(
            fn=lambda: ("", "<h3>RAG Answer</h3>", "<h3>Top Documents</h3>"),
            inputs=[],
            outputs=[question, rag_output, docs_out]
        )

    demo.launch(server_port=server_port, server_name="0.0.0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for RAG Demo Service Deployment.")
    parser.add_argument('-p','--port_no', help='Port number for the service', type=int, default=10000)
    args = parser.parse_args()
    run_gradio_app(server_port=args.port_no)
