import guardrail_models
import dataset_testing
import sys

datasets = {"XSTest": dataset_testing.XSTest, "MedSafetyBench": dataset_testing.MedSafetyBench, "All": benchmarkAll}

def benchmarkAll():
    print("--- [Benchmarking on all datasets] ---")
    dataset_testing.MedSafetyBench()
    dataset_testing.XSTest()

    return

def main():
    input_dataset = sys.argv[1]
    print("Starting benchmarks...")

    datasets[input_dataset]()


    print("End")

main()
