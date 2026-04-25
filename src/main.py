import guardrail_models
import dataset_testing

def main():
    print("Starting benchamrks...")
    dataset_testing.MedSafetyBench()

    dataset_testing.XSTest()
    print("End")

main()
