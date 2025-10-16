import json
from unitxt import get_logger
from unitxt.api import create_dataset, evaluate

logger = get_logger()

# Set up question answer pairs in a dictionary
with open('./gt_data_BWI_Docs_V1_4142bff4928f13415914eedf6627ca39.json', 'r') as f:
    dataset = json.load(f)

with open('./pred_data_BWI_Docs_V1_4142bff4928f13415914eedf6627ca39.json', 'r') as f:
    predictions = json.load(f)

# select recommended metrics according to your available resources.
metrics = [
    "metrics.rag.end_to_end.recommended.cpu_only.all",
    # "metrics.rag.end_to_end.recommended.small_llm.all",
    # "metrics.rag.end_to_end.recommended.llmaj_watsonx.all",
    # "metrics.rag.end_to_end.recommended.llmaj_rits.all"
    # "metrics.rag.end_to_end.recommended.llmaj_azure.all"
]

dataset = create_dataset(
    task="tasks.rag.end_to_end",
    test_set=dataset,
    split="test",
    postprocessors=[],
    metrics=metrics,
)

results = evaluate(predictions, dataset)

# Print Results:

print("Global Results:")
print(results.global_scores.summary)

# print("Instance Results:")
# print(results.instance_scores.summary)