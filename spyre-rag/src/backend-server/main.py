import base64
import json

from flask import Flask, request, jsonify, Response, stream_with_context
import os
import time
import requests
import sys

def read_required_env_variables(*var_names):
    missing_vars = [var for var in var_names if os.getenv(var) is None]

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return {var: os.getenv(var) for var in var_names}


try:
    env_vars = read_required_env_variables('MODEL_NAME', 'INFERENCE_SERVICE_URL', 'RETRIEVER_SERVICE_URL')
except EnvironmentError as e:
    print(e)
    sys.exit(1)

app = Flask(__name__)

def get_response_from_llm(prompt):

    headers = {
        "accept": "application/json",
        "RITS_API_KEY": "",
        "Content-type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "model": env_vars['MODEL_NAME'],
        "max_tokens": 512,
        "repetition_penalty": 1.1,
        "temperature": 0.0,
        "stream": False
    }

    try:
        endpoint = f"{env_vars['INFERENCE_SERVICE_URL']}"
        start_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        end_time = time.time()
        request_time = end_time - start_time
        data = response.json()

        return data, request_time

    except Exception as e:
        print(f"LLM service error: {e}")
        return ""

def get_response_from_db_retrieval_service(prompt):
    try:
        response = requests.post(
            f"{env_vars['RETRIEVER_SERVICE_URL']}/retrieve",
            json={"query": prompt},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        print("DATA FROM RETRIEVAL SERVICE")
        print(data)

        # Combine all page_content sections into a single context block
        documents = data.get("results", [])
        context_blocks = [doc.get("page_content", "") for doc in documents]
        return "\n\n".join(context_blocks), data
    except Exception as e:
        print(f"Retrieval service error: {e}")
        return ""

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    prompt_template = """
            You are an IT-Chatbot to help people working in IT-Operations. Answer the question strictly based on the context.
            If the CONTEXT does not answer the QUESTION then you continue generating the answer from your prior knowledge. Your answer should not contain more than 200 words.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            Answer:
            """
    context, db_response = get_response_from_db_retrieval_service(prompt)
    prompt = prompt_template.format(context=context, question=prompt)
    result, request_time = get_response_from_llm(prompt)
    print("RESULT====================",result)
    response_text = result["response"].strip()
    return jsonify({"response": response_text, "documents": db_response, "request time": request_time})

def stream_ollama(prompt):
    headers = {
        "accept": "application/json",
        "RITS_API_KEY": "",
        "Content-type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "model": env_vars['MODEL_NAME'],
        "max_tokens": 512,
        "repetition_penalty": 1.1,
        "temperature": 0.0,
        "stream": True
    }

    endpoint = f"{env_vars['INFERENCE_SERVICE_URL']}"
    with requests.post(endpoint, json=payload, headers=headers, stream=True) as r:
        for line in r.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    pass  # ignore malformed lines


@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json()
    prompt = data.get("prompt", "")
    prompt_template = """
                Answer the question strictly based on the context.
                If the context does not answer the question then you continue generating the answer from your prior knowledge.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
    context, db_response = get_response_from_db_retrieval_service(prompt)
    prompt = prompt_template.format(context=context, question=prompt)
    return Response(stream_with_context(stream_ollama(prompt)), content_type='text/plain',
                    mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type'
        })

@app.route("/image", methods=["GET"])
def get_image_base64():
    image_name = request.args.get("name")
    if len(image_name.split("/")) > 1:
        # only image name should be sent, no path
        return jsonify({"error": "Access denied"}), 403
    if not image_name:
        return jsonify({"error": "Missing 'name' query parameter"}), 400

    rel_path = "../rag_on_power/images_BWI_Docs_V1_4142bff4928f13415914eedf6627ca39"

    img_path = os.path.join(rel_path, image_name)

    try:
        with open(img_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
            return jsonify({
                "filename": os.path.basename(image_name),
                "mime_type": "image/png",
                "base64": encoded
            })
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/reference", methods=["POST"])
def get_reference_docs():
    data = request.get_json()
    prompt = data.get("prompt", "")
    context, db_response = get_response_from_db_retrieval_service(prompt)
    docs = db_response.get('results')
    for doc in docs:
        if doc['type'] == "image":
            image_name = doc['source'].split("/")[1]
            rel_path = "../rag_on_power/images_BWI_Docs_V1_4142bff4928f13415914eedf6627ca39"
            img_path = os.path.join(rel_path, image_name)

            try:
                with open(img_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as ex:
                 encoded = "image could not be fetched"
            doc['image_base_64'] = encoded
    db_response['results'] = docs
    return jsonify({"documents": db_response})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
