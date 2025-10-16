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


def classify_image_with_vlm(image_uri, model_id, client, hosting_type, retries=3):
    prompt_text = "The image is a part of a technical document. Is the image a COBOL Grammar Diagram? Answer yes or no."

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
                    max_tokens=3,
                    temperature=0
                )
                return "yes" in response.choices[0].message.content.strip()

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
                        "num_predict": 3  # Equivalent to max_tokens
                    },
                    stream=False,
                )
                return "yes" in response["message"]["content"].strip()

            else:
                raise ValueError(f"Unsupported VLM type: {hosting_type}")

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error during image classification: {e}")
            time.sleep(1)


def generate_image_summary_helper(image_uri, caption, model_id, client, hosting_type, retries=3, grammar_image=False):
    prompt_text = """
The image is part of a agricultar related document.
Be specific about visual elements. Describe the image in detail.
"""

    if caption.strip():
        prompt_text += f"The cpation of the image is {caption}"

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
        client = None
    else:
        raise ValueError(f"Unsupported hosting type: {hosting_type}")

    def process_image(image_uri, caption):
        # First classify the image
        # is_grammar_image = classify_image_with_vlm(
        #     image_uri=image_uri,
        #     model_id=vlm_model,
        #     client=client,
        #     hosting_type=hosting_type
        # )

        # Then generate summary based on classification
        return generate_image_summary_helper(
            image_uri=image_uri,
            caption=caption,
            model_id=vlm_model,
            client=client,
            hosting_type=hosting_type,
            grammar_image=False
        )

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(image_info_list)))) as executor:
        futures = [
            executor.submit(process_image, image_uri, caption)
            for image_uri, caption in image_info_list
        ]
        return [f.result() for f in futures]
