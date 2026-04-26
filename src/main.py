import guardrail_models
import dataset_testing

def main():
    print("Starting benchmarks...")
    dataset_testing.MedSafetyBench()

    dataset_testing.XSTest()
    print("End")

main()
