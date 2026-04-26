# Dataset testing
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

models = {
    "LlamaGuard": {
        "setup": guardrail_models.LlamaGuardSetup,
        "inference": guardrail_models.LlamaGuard3_1B
    },
    "GraniteGuardian": {
        "setup": guardrail_models.GraniteGuardianSetup,
        "inference": lambda prompt, model, tokenizer: guardrail_models.GraniteGuardian("user", prompt, model, tokenizer)
    },
    "ShieldGemma": {
        "setup": guardrail_models.ShieldGemmaSetup,
        "inference": guardrail_models.ShieldGemma
    }
}


def MedSafetyBench():
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
    open(f"results/{file_name}.json", "w").close()  # Create an empty .json file that eventually overwrites old results

    for i in range(msb.shape[0]):
    #for i in range(10):
        prompt = msb.iloc[i]['harmful_medical_request']
        results[i] = {"Input": prompt, "Ground_truth": "unsafe"}

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

    with open(f"results/{file_name}.json", "a") as f:
        for i in results:
            json.dump(results[i], f)
            f.write('\n')
    # files.download(f"{file_name}.json")    # Download results


def XSTest():
    xstest = load_dataset("walledai/XSTest", split="test")
    # print(xstest.features)
    # print(xstest[0])
    # print(xstest[0]['prompt'])

    # Setup for result load
    results = {}
    file_name = "XSTest_results"
    open(f"results/{file_name}.json", "w").close()  # Create an empty .json file that eventually overwrites old results

    for i in range(xstest.shape[0]):
    #for i in range(10):
        prompt = xstest[i]['prompt']
        results[i] = {"Input": prompt, "Ground_truth": xstest[i]['label']}

    for model_name, funcs in models.items():
        print(f"Benchmarking {model_name} on XSTest...\n")
        model, tokenizer = funcs["setup"]()
         for i in range(xstest.shape[0]):
        #for i in range(10):
            prompt = xstest[i]['prompt']
            output, prob_of_risk = funcs["inference"](prompt, model, tokenizer)
            results[i][f"{model_name}Output"] = output
            results[i][f"{model_name}Prob"] = round(prob_of_risk, 3)
        guardrail_models.cleanup(model, tokenizer)
        print("Done")

    with open(f"results/{file_name}.json", "a") as f:
        for i in results:
            json.dump(results[i], f)
            f.write('\n')
    # files.download(f"{file_name}.json")    # Download results


# MedSafetyBench()

# XSTest()
