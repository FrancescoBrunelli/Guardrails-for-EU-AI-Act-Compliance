import guardrail_models
import dataset_testing
import sys

def benchmarkAll():
    print("--- [Benchmarking on all datasets] ---")
    dataset_testing.MedSafetyBench()
    dataset_testing.XSTest()
    return

datasets = {"XSTest": dataset_testing.XSTest, "MedSafetyBench": dataset_testing.MedSafetyBench, "All": benchmarkAll, "Test": None}

def main():
    input_dataset = sys.argv[1]
    print("Starting benchmarks...")

    if input_dataset == "Test":
        print("Starting Test")

        guardrail_models.main()

        print("Test ended")
        return
    elif input_dataset not in datasets:
        print("Input dataset not supported")
    else: datasets[input_dataset]()

    print("End")

    return

main()
