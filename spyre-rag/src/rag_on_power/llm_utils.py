import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


with open("api_key.txt", "r") as file:
    api_key = file.read().strip()


def classify_text_with_llm(text_blocks, gen_model, gen_endpoint, batch_size=128):
    prompt_template = """You are a smart assistant for RAG corpus creation. Your task is to decide whether the following text be included in a technical documentation knowledge base? Respond only with "yes" or "no".

    "yes" for technical content: concepts, configs, commands, behavior, features that are useful for technical question answering.
    "no" for non-technical content: about a person (author/editor), acknowledgements, titles, contact info, copyright, trademark, foreword, notices, disclaimer, preface, etc. that are not useful for technical question answering.

    Text: {text}

    Answer:
    """

    all_prompts = [prompt_template.format(text=item.strip()) for item in text_blocks]
    
    decisions = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Classifying Text with LLM"):
        batch_prompts = all_prompts[i:i + batch_size]

        payload = {
            "model": gen_model,
            "prompt": batch_prompts,
            "temperature": 0,
            "max_tokens": 3,
        }
        try:
            response = requests.post(gen_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])
            for choice in choices:
                reply = choice.get("text", "").strip().lower()
                decisions.append("yes" in reply)
        except Exception as e:
            print(f"Error in vLLM: {e}")
            decisions.append(True)
    return decisions


# def classify_single_prompt(prompt, gen_model, gen_endpoint, retries=3):
#     payload = {
#         "model": gen_model,
#         "prompt": prompt,
#         "temperature": 0,
#         "max_tokens": 3,
#         "stream": False,
#     }
#     for attempt in range(retries):
#         try:
#             response = requests.post(gen_endpoint, json=payload)
#             response.raise_for_status()
#             result = response.json()
#             # try:
#             #     reply = result.get("response", "").strip().lower()
#             # except:
#             #     reply = result.get("choices", [{}])[0].get("text", "").strip().lower()
#             reply = result.get("choices", [{}])[0].get("text", "").strip().lower()
#             # reply = result.get("response", "").strip().lower()
#             return "yes" in reply
#         except Exception as e:
#             print(f"[Attempt {attempt+1}] Error in LLM call: {e}")
#             time.sleep(1)
#     return False

# def classify_text_with_llm(text_blocks, gen_model, gen_endpoint, max_workers=64):
#     prompt_template = """You are a smart assistant for RAG corpus creation. 
#     Your task is to decide whether the following text should be included in a technical documentation knowledge base. 
#     Respond only with "yes" or "no".

#     "yes" for technical content: concepts, configs, commands, behavior, features.
#     "no" for non-technical content: about a person (author/editor), acknowledgements, titles, contact info, copyright, trademark, foreword, notices, disclaimer, preface, etc.

#     Text: {text}

#     Answer:

#     """

#     all_prompts = [prompt_template.format(text=item.strip()) for item in text_blocks]

#     decisions = [None] * len(all_prompts)
#     with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(all_prompts)))) as executor:
#         futures = {
#             executor.submit(classify_single_prompt, prompt, gen_model, gen_endpoint): idx
#             for idx, prompt in enumerate(all_prompts)
#         }

#         for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying Text"):
#             idx = futures[future]
#             try:
#                 decisions[idx] = future.result()
#             except Exception as e:
#                 print(f"Thread failed at index {idx}: {e}")
#                 decisions[idx] = False
#     return decisions

def filter_with_llm(text_blocks, gen_model, gen_endpoint):
    text_contents = [block.get('text') for block in text_blocks]

    # Run classification
    decisions = classify_text_with_llm(text_contents, gen_model, gen_endpoint)
    print(f"[Debug] Prompts: {len(text_contents)}, Decisions: {len(decisions)}")
    filtered_blocks = [block for dcsn, block in zip(decisions, text_blocks) if dcsn]
    print(f"[Debug] Filtered Blocks: {len(filtered_blocks)}, True Decisions: {sum(decisions)}")
    return filtered_blocks



# def summarize_table(table_html, table_caption, gen_model, gen_endpoint, batch_size=64):
#     prompt_template = """You are a smart assistant analyzing technical documents. 
#     You are given a table extracted from a document. Your task is to summarize the key points and insights from the table. Avoid repeating the entire content; focus on what is meaningful or important. 
#     The caption of the table: {caption}

#     Table:
#     {content}

#     Summary:"""
    
#     all_prompts = [prompt_template.format(content=html, caption=caption) for html, caption in zip(table_html, table_caption)]
    
#     summaries = []
#     for i in tqdm(range(0, len(all_prompts), batch_size), desc="Summarizing Tables with LLM"):
#         batch_prompts = all_prompts[i:i + batch_size]

#         payload = {
#             "model": gen_model,
#             "prompt": batch_prompts,
#             "temperature": 0,
#             "repetition_penalty": 1.1,
#             "max_tokens": 512,
#         }
#         try:
#             response = requests.post(gen_endpoint, json=payload)
#             response.raise_for_status()
#             result = response.json()
#             choices = result.get("choices", [])
#             for choice in choices:
#                 reply = choice.get("text", "").strip().lower()
#                 summaries.append(reply)
#         except Exception as e:
#             print(f"Error summarizing table with LLM: {e}")
#             summaries.append('No summary.')
#     return summaries

def summarize_single_table(prompt, gen_model, gen_endpoint):
    payload = {
        "model": gen_model,
        "prompt": prompt,
        "temperature": 0,
        "repetition_penalty": 1.1,
        "max_tokens": 512,
        "stream": False,
    }
    try:
        response = requests.post(gen_endpoint, json=payload)
        response.raise_for_status()
        result = response.json()
        # try:
        #     reply = result.get("response").strip().lower()
        # except:
        #     reply = result.get("choices", [{}])[0].get("text", "").strip()
        reply = result.get("choices", [{}])[0].get("text", "").strip()
        return reply
    except Exception as e:
        print(f"Error summarizing table: {e}")
        return "No summary."

def summarize_table(table_html, table_caption, gen_model, gen_endpoint, max_workers=32):
    prompt_template = """You are a smart assistant analyzing technical documents. 
You are given a table extracted from a document. Your task is to summarize the key points and insights from the table. Avoid repeating the entire content; focus on what is meaningful or important. 
The caption of the table: {caption}

Table:
{content}

Summary:"""
    
    all_prompts = [
        prompt_template.format(content=html, caption=caption)
        for html, caption in zip(table_html, table_caption)
    ]

    summaries = [None] * len(all_prompts)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(all_prompts)))) as executor:
        futures = {
            executor.submit(summarize_single_table, prompt, gen_model, gen_endpoint): idx
            for idx, prompt in enumerate(all_prompts)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing Tables"):
            idx = futures[future]
            try:
                summaries[idx] = future.result()
            except Exception as e:
                print(f"Thread failed at index {idx}: {e}")
                summaries[idx] = "No summary."

    return summaries


def query_vllm(question, documents, endpoint, ckpt, language, stop_words, rag=True):
    context = "\n\n".join([doc.get("page_content") for doc in documents])
    prompt_template = """
    Answer the question strictly based on the context.
    If the context does not answer the question then you continue generating the answer from your prior knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """ if rag else """
    Question:
    {question}

    Answer:
    """
    
    if language.lower() == "english":
        question += "\n\nAnswer the question in English language."
    elif language.lower() == "german":
        question += "\n\nAnswer the question in German language."

    prompt = prompt_template.format(context=context, question=question)
    
    headers = {
        "accept": "application/json",
        "RITS_API_KEY": api_key,
        "Content-type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "model": ckpt,
        "max_tokens": 512,
        "repetition_penalty": 1.1,
        "temperature": 0.0,
        "stop": stop_words,
        "stream": False
    }
    
    try:
        start_time = time.time()
        # Use requests for synchronous HTTP requests
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        end_time = time.time()
        request_time = end_time - start_time
        return response_data, request_time
    except Exception as e:
        return {"error": str(e)}, 0.


def generate_qa_pairs(records, gen_model, gen_endpoint, batch_size=32):

    prompt_template = (
        "You are a helpful assistant creating question-answer pairs for a Retrieval-Augmented Generation (RAG) dataset.\n"
        "Given the following passage, generate **one** question that can be answered strictly using the passage content.\n"
        "Then provide the correct answer from the passage.\n\n"
        "Format your response exactly as:\n"
        "Q: <question>\n"
        "A: <answer>\n\n"

        "Reference Passage:\n{text}\n\n"
        "Q:"
    )

    all_prompts = []
    for r in records:
        prompt = prompt_template.format(text=r.get("page_content"))
        all_prompts.append(prompt)

    qa_pairs = []

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating QA Pairs"):
        batch_prompts = all_prompts[i:i+batch_size]

        payload = {
            "model": gen_model,
            "prompt": batch_prompts,
            "temperature": 0.0,
            "max_tokens": 512
        }

        try:
            response = requests.post(gen_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])

            for j, choice in enumerate(choices):
                text = choice.get("text", "").strip()
                if "Q:" in batch_prompts[j]:
                    # Try to split into question and answer
                    parts = text.split("A:", 1)
                    question = parts[0].strip().lstrip("Q:").strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": records[i + j].get("page_content", ""),
                        "chunk_id": records[i + j].get("chunk_id", "")
                    })

        except Exception as e:
            print(f"❌ Error generating QA batch: {e}")

    return qa_pairs
