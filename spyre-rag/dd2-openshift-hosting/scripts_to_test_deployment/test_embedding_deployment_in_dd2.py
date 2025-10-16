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
    prompt = "Write a poem on weather."
    #ckpt = "ibm-granite/granite-embedding-125m-english"
    ckpt = "ibm-granite/granite-embedding-278m-multilingual"

    payload ={
            "model": ckpt, 
            "input": ["""evaluated on benchmarks. Prior work has found systematic relationships between web corpus loss
and benchmark performance (Wei et al., 2022; Huang et al., 2024), which suggests the possibility of
using correlations between perplexity and benchmark scores as the basis for a data selection policy.
In the present paper, we pursue this possibility and find a radically simple approach that is also
effective: we select data via perplexity correlations (Figure 1), where we select data domains (e.g.
wikipedia.org, stackoverflow.com, etc.) for which LLM log-probabilities are highly correlated with
downstream benchmark performance. To enable our approach, we complement our algorithm with
a statistical framework for correlation-based data selection and derive correlation estimators that
perform well over our heterogeneous collection of LLMs.
We validate our approach using a collection of pretrained causal LLMs on the Hugging Face Open
LLM Leaderboard (Beeching et al., 2023) and find that perplexity correlations are predictive of an
LLM’s benchmark performance. Importantly, we find that these relationships are robust enough to
enable reliable data selection that targets downstream benchmarks. In controlled pretraining experiments at the 160M parameter scale on eight benchmarks, our approach strongly outperforms DSIR
(Xie et al., 2023b) (a popular training-free data selection approach based on n-gram statistics) while
generally matching the performance of the best method validated at scale by Li et al. (the OH-2.5
+ELI5 fastText classifier; Joulin et al. 2016) without any parameter tuning or human curation. In
followup experiments at the 160M to 1.4B parameter scale which we pre-registered, our approach
outperforms the best Li et al. filter on the main benchmark from their paper (an aggregate of 22
benchmarks) when filtering from their base data pool, and both approaches remain close to each
other when filtering from their extensively pre-filtered pool. We further find that the performance of
our approach strengthens with increasing scale.
2 RELATED WORK
To go beyond the status quo of deduplication, perplexity filtering, and hand-curation (Laurençon
et al., 2022; BigScience, 2023; Marion et al., 2023; Abbas et al., 2023; Groeneveld et al., 2024;
Soldaini et al., 2024; Penedo et al., 2024; Llama Team, 2024), targeted methods have been proposed
to filter pretraining data so that the resulting LLM will achieve higher scores on given benchmarks.
There are lightweight approaches that use n-gram overlap (Xie et al., 2023b) or embedding similarity
(Everaert & Potts, 2024) to select training data that is similar to data from a given benchmark. There
are also less-scalable methods that require training proxy LLMs on different data mixtures (Ilyas
et al., 2022; Xie et al., 2023a; Engstrom et al., 2024; Liu et al., 2024; Llama Team, 2024).
Given the high costs of proxy-based data selection methods, they have primarily been used to select
among human-curated pretraining data mixtures (Llama Team, 2024; Li et al., 2024) rather than
a high dimensional space of mixtures. Our work takes an orthogonal approach and builds upon
recent observational studies that have found scaling relationships that hold across collections of
uncontrolled and diverse LLMs (Owen, 2024; Ruan et al., 2024). While these studies do not examine"""],
            "temperature": 0,
            "truncate_prompt_tokens": 512
            }


    #endpoint = "https://granite-embed-english-spyre-showcase-dev.apps.dev.spyre.res.ibm.com/v1/embeddings"
    endpoint = "https://granite-embed-multi-spyre-showcase-dev.apps.dev.spyre.res.ibm.com/v1/embeddings"


    response = dd2_vllm_inference(payload, endpoint)
    print("\n\n================Printing response:==================")
    print(response.json())# prints the entire response
    #print(response.json()["data"][0]["embedding"]) ## Prints only embedding 
    print("====================================================")
    print("====================================================\n")
    end_time = time.time()
    print("Total time taken in seconds:", end_time-start_time)
    print("\n====================================================\n")

if __name__ == "__main__":
    main()

