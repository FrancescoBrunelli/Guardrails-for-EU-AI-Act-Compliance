# Results
This directory contains all the raw results collected from the experiments and their analyzed version. The `.json` files are structured in the following manner:

#### Raw data files:
Each raw result file, named as `<dataset_name>_results.json`, contains:
- Input prompt
- Output response (e.g., for the ExpGuardMix dataset)
- Ground_truth (either for the input or, for the ExpGuardMix dataset both input and response)
- Category (when available)
- For each guardrail model evaluated:
	- <model_name>Output: either safe or unsafe
	- <model_name>Prob: probability score associated with the output provided by the model
	- <model_name>Time: inference time in seconds

#### Analyzed data files:
The `analyzed_output/` folder contains aggregated evaluation metrics computed from the raw results.

There are three types of analyzed outputs:
1. **Single-model analysis files**  
	- Files named with `(single)` contain metrics computed only for individual guardrail models.  
	- Example:  
		- `XSTest(single)_score_results.json`  
  
2. **Combined analysis files**  
	- Files without `(single)` include both:  
		- Individual model evaluations  
		- Ensemble/combined model evaluations using `AND` / `OR` / `AVG` logic  
	- Example:  
		- `XSTest_score_results.json`
  
3. **Category-level analysis files**
	- Located inside the `Categories/` folder, these files contain per-category evaluation metrics.  
	- These analyses are generated only for single-model evaluations.  
	- Example:  
		- `XSTestCategories/`

Each analyzed entry contains:
- Compressed model name (see [legend](../README.md#legend))
- Dataset used
- Number of models combined together (`1` for single-model evaluation)
- Ensemble logic: `AND` / `OR` / `AVG` (`null` for single models)
- Number of total predictions made
- Average execution time (only for single models; `null` for combined evaluations)
- Number of True Positives (TP)
- Number of False Positives (FP)
- Number of True Negatives (TN)
- Number of False Negatives (FN)
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Accuracy
- Specificity
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

**Note:** some metrics may be N/A depending on the dataset used (see [datasets](../src/README.md#datasets))

---
## Directory Structure

``` text
results
├── README.md
├── ExpGuardMix_InputOutput_results.json
├── ExpGuardMix_Input_results.json
├── MedSafetyBench_results.json
├── PatientSafetyBench_results.json
├── XSTest_results.json
└── analysis_output
    ├── ExpGuardMix_Input(single)_score_results.json
    ├── ExpGuardMix_InputOutput(single)_score_results.json
    ├── ExpGuardMix_InputOutput_score_results.json
    ├── ExpGuardMix_Input_score_results.json
    ├── MedSafetyBench(single)_score_results.json
    ├── MedSafetyBench_score_results.json
    ├── PatientSafetyBench(single)_score_results.json
    ├── PatientSafetyBench_score_results.json
    ├── XSTest(single)_score_results.json
    ├── XSTest_score_results.json
    ├── ExpGuardMix_InputCategories/
    ├── ExpGuardMix_InputOutputCategories/
    ├── PatientSafetyBenchCategories/
    └── XSTestCategories/
```
 