import json
import requests
import numpy as np


with open("api_key.txt", "r") as file:
    api_key = file.read().strip()


class FastAPIEmbeddingFunction:
    def __init__(self, emb_model, emb_endpoint, max_tokens):
        self.emb_model = emb_model
        self.emb_endpoint = emb_endpoint
        self.max_tokens = max_tokens

    def embed_documents(self, texts):
        return self._call_fastapi_embedding(texts)

    def embed_query(self, text):
        return self._call_fastapi_embedding([text])[0]

    def _call_fastapi_embedding(self, texts):
        payload = {
            "input": texts,
            "model": self.emb_model,
            "truncate_prompt_tokens": self.max_tokens-1,
        }
        headers = {
            "accept": "application/json",
            "RITS_API_KEY": api_key,
            "Content-type": "application/json"
        }
        response = requests.post(
            self.emb_endpoint,
            data=json.dumps(payload),
            headers=headers
        )
        response.raise_for_status()
        r = response.json()
        embeddings = [data['embedding'] for data in r['data']]
        return [np.array(embed, dtype=np.float32) for embed in embeddings]
