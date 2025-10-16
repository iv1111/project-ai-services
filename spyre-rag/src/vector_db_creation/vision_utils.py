import time
import base64
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from ollama import chat as ollama_chat

import base64
from PIL import Image
from io import BytesIO
import tempfile



def create_temp_image(b64_data):
    # Convert generator to string if needed
    if not isinstance(b64_data, str):
        b64_data = ''.join(b64_data)

    # Check for comma separator in data URI
    if "," not in b64_data:
        raise ValueError("Expected data URI format: 'data:image/...;base64,...'")

    try:
        img_data = base64.b64decode(b64_data.split(",", 1)[1])
        pil_image = Image.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        return [tmp.name]


def generate_image_summary_helper(image_uri, caption, model_id, client, hosting_type, retries=3):
    prompt_text = (
        "The image is part of a technical document for RAG. "
        "Be specific about visual elements like diagrams, tables, bar graphs, or labels. "
        f"Describe it in detail. Its Caption: {caption}."
    )
    for attempt in range(retries):
        try:
            if hosting_type == "vllm":
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_uri}},
                                {"type": "text", "text": prompt_text}
                            ]
                        }
                    ],
                    max_tokens=512,
                    temperature=0
                )
                return response.choices[0].message.content.strip()

            elif hosting_type == "ollama":
                response = ollama_chat(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_text,
                            "images": create_temp_image(image_uri)
                        }
                    ],
                    options={
                        "temperature": 0.,
                        "num_predict": 512  # Equivalent to max_tokens
                    },
                    stream=False,
                )
                return response["message"]["content"].strip()

            else:
                raise ValueError(f"Unsupported VLM type: {hosting_type}")

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error during image summary: {e}")
            time.sleep(1)

    return "Image summary failed after multiple attempts."


def generate_image_summary(image_info_list, vlm_model, vlm_endpoint, hosting_type="vllm"):
    if hosting_type == "vllm":
        client = OpenAI(api_key="EMPTY", base_url=vlm_endpoint)
    elif hosting_type == "ollama":
        client = None  # Not used, since we call `ollama.chat` directly
    else:
        raise ValueError(f"Unsupported hosting type: {hosting_type}")

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(image_info_list)))) as executor:
        futures = [
            executor.submit(
                generate_image_summary_helper,
                image_uri,
                caption,
                vlm_model,
                client,
                hosting_type
            )
            for image_uri, caption in image_info_list
        ]
        return [f.result() for f in futures]




# import requests
# import time
# from concurrent.futures import ThreadPoolExecutor

# def generate_image_summary_helper(image_url, caption, model_id, endpoint, retries=3):
#     headers = {
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": model_id,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_url
#                         }
#                     },
#                     {
#                         "type": "text",
#                         "text": (
#                             "The image is part of a technical document for RAG. "
#                             "Be specific about visual elements like diagrams, tables, bar graphs, or labels. "
#                             f"Describe it in detail. Its Caption: {caption}."
#                         )
#                     }
#                 ]
#             }
#         ],
#         "max_tokens": 512,
#         "temperature": 0,
#     }

#     for attempt in range(retries):
#         try:
#             response = requests.post(endpoint, headers=headers, json=payload)
#             response.raise_for_status()
#             result = response.json()
#             return result['choices'][0]['message']['content'].strip()
#         except Exception as e:
#             print(f"[Attempt {attempt+1}] Error during image summary: {e}")
#             time.sleep(1)

#     return "Image summary failed after multiple attempts."


# def generate_image_summary(image_info_list, vlm_model, vlm_endpoint):
#     print('In vision utils')
#     print(image_info_list[0])
#     with ThreadPoolExecutor(max_workers=max(1, min(8, len(image_info_list)))) as executor:
#         futures = [
#             executor.submit(generate_image_summary_helper, img_url, caption, vlm_model, vlm_endpoint)
#             for img_url, caption in image_info_list
#         ]
#         return [f.result() for f in futures]


