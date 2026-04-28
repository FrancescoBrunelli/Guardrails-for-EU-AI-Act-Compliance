# Dataset testing
from unittest import result

from datasets import load_dataset
import os
import pandas as pd
import json
import guardrail_models
#from datasets.PolyGuard.social_media.guardrail_model import guardrail
import guardrail_models

# Patient Safety Bench (from HuggingFace)
PT_SPEECH_BASE_MODEL_SAMPLE = load_dataset("microsoft/PatientSafetyBench")
# print(psb["train"].features)
# print(psb["train"][0])

# datasets: MedSafetyBench, XSTest, PatientSafetyBench (to do), JailbreakBench (to do), FACTS-med (to do?)
#
models = {
    "LlamaGuard1B": {
        "setup": guardrail_models.LlamaGuard1BSetup,
        "inference": guardrail_models.LlamaGuardInference
    },
    "LlamaGuard8B": {
        "setup": guardrail_models.LlamaGuard8BSetup,
        "inference": guardrail_models.LlamaGuardInference
    },
    "GraniteGuardian2B": {
        "setup": guardrail_models.GraniteGuardian2BSetup,
        "inference": lambda prompt, model, tokenizer: guardrail_models.GraniteGuardian2B("user", prompt, model, tokenizer)
    },
    "ShieldGemma2B": {
        "setup": guardrail_models.ShieldGemma2BSetup,
        "inference": guardrail_models.ShieldGemma2B
    },
    "WildGuard7B": {
        "setup":guardrail_models.WildGuard7BSetup,
        "inference": guardrail_models.WildGuard7B
    }
}

def MedSafetyBench(model_name):
    # Merge test datasets into one single dataset
    path = "datasets/med-safety-bench/datasets/test/gpt4"
    msb = []
    file_list = os.listdir(path)

    for f in file_list:
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, f), index_col=0)
            msb.append(df)

    msb = pd.concat(msb, ignore_index=True)  # Dataset merged

    results = {}
    file_name = "MedSafetyBench_results"

    # Check if the file already exists. If not create and setup a new one
    results_path = f"results/{file_name}.json"
    if os.path.exists(results_path):
        print(f"{results_path} already exists; updating the existing one")
        with open(f"results/{file_name}.json", "r") as f:  # Load results stored in the .json file
            loaded = [json.loads(line) for line in f if line.strip()]
            results = {i: row for i, row in enumerate(loaded)}
    else:
        print(f"{results_path} not found; creating file")
        open(f"results/{file_name}.json", "w").close()  # Create an empty .json file
        for i in range(msb.shape[0]):
        #for i in range(10):
            prompt = msb.iloc[i]['harmful_medical_request']
            results[i] = {"Input": prompt, "Ground_truth": "unsafe"}

    '''
    for model_name, funcs in models.items():
        print(f"Benchmarking {model_name} on MedSafetyBench...\n")
        model, tokenizer = funcs["setup"]()
        for i in range(msb.shape[0]):
        #for i in range(10):
            prompt = msb.iloc[i]['harmful_medical_request']
            output, prob_of_risk = funcs["inference"](prompt, model, tokenizer)
            results[i][f"{model_name}Output"] = output
            results[i][f"{model_name}Prob"] = round(prob_of_risk, 3)
        guardrail_models.cleanup(model, tokenizer)
        print("Done\n")
    '''

    print(f"Benchmarking {model_name} on MedSafetyBench...\n")
    model, tokenizer = models[model_name]["setup"]()
    for i in range(msb.shape[0]):
    #for i in range(10):
        prompt = msb.iloc[i]['harmful_medical_request']
        output, prob_of_risk = models[model_name]["inference"](prompt, model, tokenizer)
        results[i][f"{model_name}Output"] = output
        results[i][f"{model_name}Prob"] = round(prob_of_risk, 3)

        '''
        # Test to delete
        results[i][f"{model_name}Prompt"] = prompt
        # End test to delete
        '''

    guardrail_models.cleanup(model, tokenizer)

    with open(f"results/{file_name}.json", "w") as f:
        for i in results:
            json.dump(results[i], f)
            f.write('\n')
    print("Done\n")
    # files.download(f"{file_name}.json")    # Download results
    return

def XSTest(model_name):
    xstest = load_dataset("walledai/XSTest", split="test")
    # print(xstest.features)
    # print(xstest[0])
    # print(xstest[0]['prompt'])

    results = {}
    file_name = "XSTest_results"

    # Check if the file already exists. If not create and setup a new one
    results_path = f"results/{file_name}.json"
    if os.path.exists(results_path):
        print(f"{results_path} already exists; updating the existing one")
        with open(f"results/{file_name}.json", "r") as f:  # Load results stored in the .json file
            loaded = [json.loads(line) for line in f if line.strip()]
            results = {i: row for i, row in enumerate(loaded)}
    else:
        print(f"{results_path} not found; creating file")
        open(f"results/{file_name}.json", "w").close()  # Create an empty .json file
        for i in range(xstest.shape[0]):
        #for i in range(10):
            prompt = xstest[i]['prompt']
            results[i] = {"Input": prompt, "Ground_truth": xstest[i]['label'], "Category": xstest[i]['type']}

    print(f"Benchmarking {model_name} on XSTest...\n")
    model, tokenizer = models[model_name]["setup"]()
    for i in range(xstest.shape[0]):
    #for i in range(10):
        prompt = xstest[i]['prompt']
        output, prob_of_risk = models[model_name]["inference"](prompt, model, tokenizer)
        results[i][f"{model_name}Output"] = output
        results[i][f"{model_name}Prob"] = round(prob_of_risk, 3)

        '''
        # Test to delete
        results[i][f"{model_name}Prompt"] = prompt
        # End test to delete
        '''

    guardrail_models.cleanup(model, tokenizer)

    with open(f"results/{file_name}.json", "w") as f:
        for i in results:
            json.dump(results[i], f)
            f.write('\n')
    print("Done\n")
    # files.download(f"{file_name}.json")    # Download results
    return

def PatientSafetyBench(model_name):
    psb = load_dataset("microsoft/PatientSafetyBench", split="train")
    categories = {1:"Harmful Advice", 2:"Overconfidence", 3:"Unlicensed Practice", 4:"Misinformation", 5:"Discrimination"}
    results = {}
    file_name = "PatientSafetyBench_results"

    # Check if the file already exists. If not create and setup a new one
    results_path = f"results/{file_name}.json"
    if os.path.exists(results_path):
        print(f"{results_path} already exists; updating the existing one")
        with open(f"results/{file_name}.json", "r") as f:  # Load results stored in the .json file
            loaded = [json.loads(line) for line in f if line.strip()]
            results = {i: row for i, row in enumerate(loaded)}
    else:
        print(f"{results_path} not found; creating file")
        open(f"results/{file_name}.json", "w").close()  # Create an empty .json file
        for i in range(psb.shape[0]):
        #for i in range(10):
            prompt = psb[i]['content']
            category = categories[psb[i]['category']]
            results[i] = {"Input": prompt, "Ground_truth": "unsafe", "Category": category}

    print(f"Benchmarking {model_name} on PatientSafetyBench...\n")
    model, tokenizer = models[model_name]["setup"]()

    for i in range(psb.shape[0]):
    #for i in range(10):
        prompt = psb[i]['content']
        output, prob_of_risk = models[model_name]["inference"](prompt, model, tokenizer)
        results[i][f"{model_name}Output"] = output
        results[i][f"{model_name}Prob"] = round(prob_of_risk, 3)

    guardrail_models.cleanup(model, tokenizer)
    with open(f"results/{file_name}.json", "w") as f:
        for i in results:
            json.dump(results[i], f)
            f.write('\n')
    print("Done\n")

    #results[0] = {"Input": psb[0]['content'], "Ground_truth": "unsafe", "Category": categories[psb[0]['category']]}
    #print(results)

    return

'''
def main():
    PatientSafetyBench(None)
    return

main()
'''
