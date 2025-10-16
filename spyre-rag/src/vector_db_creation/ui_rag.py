import gradio as gr
import argparse
from pymilvus import connections
from langchain_milvus import BM25BuiltInFunction, Milvus

from misc_utils import get_model_endpoints
from retrieval_utils import search_and_answer_dual
from emb_utils import FastAPIEmbeddingFunction
from db_utils import generate_collection_name


# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

OpenAI_API_KEY = "EMPTY"
TOP_K = 20
TOP_R = 5

# Default DB prefix pattern
DB_NAME_PREFIX= 'BWI_Docs_V1'
vectorstore = None
# Keep track of the last used config for vectorstore
last_vectorstore_config = {
    "emb_model_choice": None,
    "vlm_model_choice": None,
    "llm_model_choice": None
}

# Globals to be set dynamically
emb_model_dict = {}
vlm_model_dict = {}
llm_model_dict = {}
reranker_model_dict = {}


def initialize_models(deployment_type):
    global emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict
    emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict = get_model_endpoints(deployment_type)

    return (
        gr.update(choices=list(emb_model_dict.keys()), value=list(emb_model_dict.keys())[0]),
        gr.update(choices=list(vlm_model_dict.keys()), value=list(vlm_model_dict.keys())[0]),
        gr.update(choices=list(llm_model_dict.keys()), value=list(llm_model_dict.keys())[-1]),
        gr.update(choices=list(llm_model_dict.keys()), value=list(llm_model_dict.keys())[-1]),
        gr.update(choices=list(reranker_model_dict.keys()), value=list(reranker_model_dict.keys())[0]),
    )


def initialize_vectorstore_if_needed(
    emb_model_choice, vlm_model_choice, llm_model_choice, db_name_prefix,
    emb_model, emb_endpoint, max_tokens
):
    global vectorstore, last_vectorstore_config

    # Check if config has changed
    config_changed = (
        emb_model_choice != last_vectorstore_config["emb_model_choice"] or
        vlm_model_choice != last_vectorstore_config["vlm_model_choice"] or
        llm_model_choice != last_vectorstore_config["llm_model_choice"]
    )

    if vectorstore is None or config_changed:
        print("🔄 Reinitializing vectorstore due to config change...")
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        embeddings = FastAPIEmbeddingFunction(emb_model, emb_endpoint, max_tokens)
        collection_name = generate_collection_name(emb_model_choice, vlm_model_choice, llm_model_choice, db_name_prefix)

        try:
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=collection_name,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                consistency_level="Strong"
            )
            # Update the last config
            last_vectorstore_config = {
                "emb_model_choice": emb_model_choice,
                "vlm_model_choice": vlm_model_choice,
                "llm_model_choice": llm_model_choice
            }
        except Exception as e:
            print(f"❌ Error initializing vectorstore: {e}")
    else:
        print("✅ Reusing existing vectorstore.")


def wrapper_search_and_answer_dual(
    question, emb_model_choice, vlm_model_choice, llm_for_db_choice,
    llm_for_qa, reranker, language
):
    try:
        emb_model = emb_model_dict[emb_model_choice]['emb_model']
        emb_endpoint = emb_model_dict[emb_model_choice]['emb_endpoint']
        emb_max_tokens = emb_model_dict[emb_model_choice]['max_tokens']
        llm_model = llm_model_dict[llm_for_qa]['llm_model']
        llm_endpoint = llm_model_dict[llm_for_qa]['llm_endpoint']
        reranker_model = reranker_model_dict[reranker]['reranker_model']
        reranker_endpoint = reranker_model_dict[reranker]['reranker_endpoint']

        # 🔁 Initialize vectorstore if config has changed
        initialize_vectorstore_if_needed(
            emb_model_choice, vlm_model_choice, llm_for_db_choice, DB_NAME_PREFIX,
            emb_model, emb_endpoint, emb_max_tokens
        )

        stop_words = ""

        (rag_ans, no_rag_ans, docs) = search_and_answer_dual(
            question, llm_endpoint, llm_model, reranker_model, reranker_endpoint, TOP_K, TOP_R, stop_words, language, vectorstore
        )

        return rag_ans, no_rag_ans, docs

    except Exception as e:
        error_html = f"<pre style='color:red;'>Error: {repr(e)}</pre>"
        return error_html, "", ""



def run_gradio_app(server_port):
    with gr.Blocks(title="🤖 SypreRAG Demo") as demo:
        gr.Markdown("# 🤖 SypreRAG Demo (for BWI)")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Follow these steps:")
                gr.Markdown(
                    """
                    - Step 1: Select deployment type (e.g., cpu, cuda, or spyre). Click on "Commit Deployment Type."
                    - Step 2: Select models used for documents ingestion.
                    - Step 3: Select models for retrieval and generation.
                    - Step 4: Select answer language.
                    - Step 5: Type your query. Click "Search and Answer."
                    """
                )
                gr.Markdown("### Step 1: Select Deployment Type")
                deployment_type = gr.Dropdown(
                    label="Deployment Type",
                    choices=["cpu", "cuda", "spyre"],
                    value="cpu"
                )
                init_button = gr.Button("Commit Deployment Type")

            # Dropdowns populated dynamically
            with gr.Column():
                gr.Markdown("### Step 2: Select Models used for Documents Ingestion")
                emb_model_choice = gr.Dropdown(label="Embedding Model")
                vlm_model_choice = gr.Dropdown(label="Vision Model")
                llm_for_db_choice = gr.Dropdown(label="LLM")

        gr.Markdown("<br>")
        with gr.Row():
            with gr.Column():
                    gr.Markdown("### Step 3: Select Models for Retrieval and Generation")
                    with gr.Row():
                        llm_for_qa = gr.Dropdown(label="LLM for Generation")
                        reranker = gr.Dropdown(label="Reranker Model for Retrieval")
            with gr.Column():
                gr.Markdown("### Step 4: Select Answer Language")
                # stop_words = gr.Textbox(label="Stop words (comma-separated, Optional)")
                language = gr.Radio(choices=["English", "German"], label="Answer Language", value="English")
            
        gr.Markdown("<br>")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 5: Type your Query")
                question = gr.Textbox(label="Query", placeholder="Enter your query...")
        with gr.Row():
            run_button = gr.Button("Search and Answer")
            clear_button = gr.Button("Clear")

        init_button.click(
            fn=initialize_models,
            inputs=[deployment_type],
            outputs=[
                emb_model_choice,
                vlm_model_choice,
                llm_for_db_choice,
                llm_for_qa,
                reranker,
            ]
        )

        rag_out = gr.HTML(label="RAG Answer", value="<h3>RAG Answer</h3>")
        no_rag_out = gr.HTML(label="No-RAG Answer", value="<h3>No-RAG Answer</h3>")
        docs_out = gr.HTML(label="Top Documents", value="<h3>Top Documents</h3>")

        run_button.click(
            wrapper_search_and_answer_dual,
            inputs=[
                question, emb_model_choice, vlm_model_choice,
                llm_for_db_choice, llm_for_qa, reranker, language
            ],
            outputs=[rag_out, no_rag_out, docs_out]
        )

        clear_button.click(
            fn=lambda: ("", "<h3>RAG Answer</h3>", "<h3>No-RAG Answer</h3>", "<h3>Top Documents</h3>"),
            inputs=[],
            outputs=[question, rag_out, no_rag_out, docs_out]
        )

    demo.launch(server_port=server_port, server_name="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for RAG Demo Service Deployment.")
    parser.add_argument('-p','--port_no', help='Port number for the service', type=int, default=10000)
    args = parser.parse_args()
    run_gradio_app(server_port=args.port_no)
