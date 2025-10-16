import time
import base64

from llm_utils import query_vllm
from reranker_utils import rerank_documents


def format_table_html(table_html):
    """
    Ensures that the table HTML is properly formatted.
    This is a basic check to wrap the table inside a <table> tag if it isn't already wrapped.
    """
    if not table_html.startswith("<table"):
        table_html = f"<table>{table_html}</table>"
    return table_html

def show_document_content(retrieved_documents, scores):
    html_content = ""
    
    for idx, (doc, score) in enumerate(zip(retrieved_documents, scores)):
        doc_metadata = doc.metadata
        doc_type = doc_metadata.get("type")
        
        # Document Header with Score
        document_header = f'<h4>Document {idx + 1} (Score: {score:.4f}), (Doc: {doc_metadata.get("filename")})</h4>'
        html_content += document_header
        
        # If the document is an image
        if doc_type == "image":
            image_path = doc_metadata.get("source")
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            image_html = f'<div style="border: 1px solid #ccc; padding: 10px; background-color: #f0f0f0; width: 100%; margin-top: 20px;">'
            image_html += f'<img src="data:image/jpeg;base64,{encoded_string}" alt="Image {doc_metadata.get("chunk_id")}" style="width: 50%; height: auto;" />'
            image_summary = f'<p><strong>Image Summary:</strong> {doc.page_content}</p>'
            image_html += f'{image_summary}</div>'
            html_content += image_html

        # If the document is a table
        elif doc_type == "table":
            table_html = doc_metadata.get("source")
            if table_html:
                table_html = format_table_html(table_html)  # Ensure proper HTML wrapping
                table_summary = f'<p><strong>Table Summary:</strong> {doc.page_content}</p>'
                html_content += f'<div style="margin-top: 20px; border: 1px solid #ccc; padding: 10px; background-color: #f0f0f0;">{table_html}<br>{table_summary}</div>'

        # If the document is plain text
        elif doc_type == "text":
            converted_doc_string = doc.page_content.replace("\n", "<br>")
            html_content += f'<div style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; background-color: #f0f0f0;">{converted_doc_string}</div>'

    return html_content

def retrieve_documents(query, vectorstore, top_k):
    results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k, ranker_type="rrf", ranker_params={"k": 100})
    retrieved_documents, scores = zip(*results_with_scores)
    return retrieved_documents, scores

def search_and_answer_dual(
        question, llm_endpoint, llm_model, reranker_model, reranker_endpoint, top_k, top_r, stop_words, language, vectorstore
    ):
    
    # Perform retrieval
    retrieval_start = time.time()
    retrieved_documents, scores = retrieve_documents(question, vectorstore, top_k)
    reranked = rerank_documents(question, retrieved_documents, reranker_model, reranker_endpoint)
    ranked_documents = []
    ranked_scores = []
    for i, (doc, score) in enumerate(reranked, 1):
        ranked_documents.append(doc)
        ranked_scores.append(score)
        if i == top_r:
            break
    retrieval_end = time.time()
    
    # Prepare stop words
    if stop_words:
        stop_words = stop_words.strip(' ').split(',')
        stop_words = [w.strip() for w in stop_words]
        stop_words = list(set(stop_words) + set(['### Response:', 'Answer:', '### Instruction:', 'Input:']))
    else:
        stop_words = ['### Response:', 'Answer:', '### Instruction:', 'Input:']
    
    # Call show_document_content to format retrieved documents
    html_content = show_document_content(ranked_documents, ranked_scores)
    
    # RAG Answer Generation
    rag_answer, rag_generation_time = query_vllm(
        question, ranked_documents, llm_endpoint, llm_model, language, stop_words, rag=True
    )
    
    # No-RAG Answer Generation
    no_rag_answer, no_rag_generation_time = query_vllm(
        question, [], llm_endpoint, llm_model, language, stop_words, rag=False
    )
    
    rag_text = rag_answer.get('choices', [{}])[0].get('text', 'No RAG answer generated.')
    no_rag_text = no_rag_answer.get('choices', [{}])[0].get('text', 'No No-RAG answer generated.')

    if rag_text == 'No RAG answer generated.':
        rag_text = rag_answer.get('response', 'No RAG answer generated.')
        no_rag_text = no_rag_answer.get('response', 'No No-RAG answer generated.')
    
    return (
        f"<h3>RAG Answer (Generation Time - {rag_generation_time:.2f} seconds):</h3><p>{rag_text}</p>",
        f"<h3>No-RAG Answer (Generation Time - {no_rag_generation_time:.2f} seconds):</h3><p>{no_rag_text}</p>",
        f"<h3>Top Documents (Retrieval and Reranking Time - {retrieval_end - retrieval_start:.2f} seconds):</h3>{html_content}",
    )