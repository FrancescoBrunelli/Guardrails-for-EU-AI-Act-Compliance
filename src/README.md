# Src

## Table of Contents
- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Main Components](#main-components)
- [Compatibility](#compatibility)
- [Notes](#notes)
- [Example Usage](#example-usage)

---

## Introduction
This directory contains the Python scripts for benchmarking and analyzing the selected guardrail models on four different datasets.
## Directory Structure

``` text
src
├── README.md
├── dataset_testing.py
├── guardrail_models.py
├── main.py
└── result_analysis.py
```

## Main Components
### `main.py`
Given the name of the guardrail model and the dataset to benchmark on, it calls the correct benchmark function specifying the model to benchmark

### `guardrail_models.py`
Script responsible for setting up and running inference of the selected guardrail models. For each model, the script contains:
- A setup function (e.g., `LlamaGuard8BSetup()`) responsible for loading the model's weights and returning the model and its tokenizer
- An inference function (e.g., `LlamaGuard8B(prompt, model, tokenizer, response = None)`) responsible for running inference of the model, providing either input or input and response, and returning the model's label response (as either `safe` / `unsafe`) and the associated probability score
- Additional relevant functions necessary for either running the model or calculating the model's response probability score

### `dataset_testing.py`
Script responsible to benchmark a given guardrail model on the specified dataset. Particularly, the script manages:
- Dataset loading (either from HuggingFace or from the cloned GitHub folder)
- Checking if there's already an existing `.json` file for the dataset's results, eventually creating it (otherwise it loads in a variable its content results)
- For each entry in the dataset, it passes it to the model and gathers all results
- Overwrites the `.json` file with the updated results

### `result_analysis.py`
For each dataset, analyzes the respective raw result file, creates the `analysis_output/` folder (inside the `results/`), calculates all previously mentioned metrics and saves them inside a `.json` file inside the folder.

	Usage: `python3 src/result_analysis.py`

## Compatibility
In order to run the benchmarks, the program needs to receive as input the name of the dataset to benchmark on and the name of the model that has to be benchmarked.

A list of available datasets and models, along with their CLI identifiers, is provided below.
### Datasets

| Dataset Name       | CLI Name |
| ------------------ | -------- |
| XSTest             | xstest   |
| MedSafetyBench     | msb      |
| PatientSafetyBench | psb      |
| ExpGuardMix        | egm      |
### Models

| Model           | Parameters | CLI Name          |
| --------------- | ---------- | ----------------- |
| LlamaGuard      | 1B         | LlamaGuard1B      |
| GraniteGuardian | 2B         | GraniteGuardian2B |
| ShieldGemma     | 2B         | ShieldGemma2B     |
| WildGuard       | 7B         | WildGuard7B       |
| LlamaGuard      | 8B         | LlamaGuard8B      |
**Note:** These identifiers are used as CLI arguments in `main.py`
## Notes
- It is recommended to run all scripts from the project root directory
- **Outputs:** a `.json` format inside the `results/` folder containing, for each dataset's entry
	-  Input prompt
	- Output response (e.g., for the ExpGuardMix dataset)
	- Ground_truth (either for the input or, for the ExpGuardMix dataset both input and response)
	- Category (when available)
	- For each guardrail model evaluated:
		- <model_name>Output: either safe or unsafe
		- <model_name>Prob: probability score associated with the output provided by the model
		- <model_name>Time: inference time in seconds

**Note:** benchmarks have been run with the following versions:
- PyTorch: `2.7.1 (CUDA 11.8)`
- Transformers: `4.57.6`
## Example Usage
``` sh
# Example run benchmarking LlamaGuard8B on XSTest dataset
python3 src/main.py xstest LlamaGuard8B
```

