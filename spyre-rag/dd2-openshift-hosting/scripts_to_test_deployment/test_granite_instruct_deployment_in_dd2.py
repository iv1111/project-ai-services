import json
import requests
import time
def dd2_vllm_inference(payload, endpoint):
    headers = {"Content-type": "application/json"}
    data_json = json.dumps(payload)
    try:
        response = requests.post(endpoint, data=data_json, headers=headers)
        return response
    except Exception as e:
        print(f"vllm Exception : {e}")
        return None


def main():
    start_time = time.time()
    f = open("/Users/saswatidana/MY_HOME/Year_2025/sample_COBOL_PLI_examples/cobol_benchmark_prompt_3.txt", "r")
    prompt = f.read() #"Write a paragraph about invention of electric bulb."
    ckpt = "ibm-granite/granite-3.3-8b-instruct"
    payload = {
            "prompt" : prompt, 
            "model" : ckpt,
            "max_tokens": 1024, #512
            "temperature": 0, 
            "truncate_prompt_tokens": 4096 #2048
    	
        }


    endpoint ="https://granite-8b-instruct-spyre-showcase-dev.apps.dev.spyre.res.ibm.com/v1/completions"
    #endpoint ="https://granite-8b-instruct-4c-spyre-showcase-dev.apps.dev.spyre.res.ibm.com/v1/completions"

    response = dd2_vllm_inference(payload, endpoint)
    print("\n\n================Printing response:==================")
    print(response.json())#['choices'][0]['text'])
    print("====================================================")
    print("====================================================\n")
    end_time = time.time()
    print("Total time taken in seconds:", end_time-start_time)
    print("\n====================================================\n")

if __name__ == "__main__":
    main()

