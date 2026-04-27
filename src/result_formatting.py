import os
import pandas as pd
import json
from tabulate import tabulate

headers1 = ['Model', 'Logic', 'TP', 'TN', 'FP', 'FN', 'FPR', 'FNR']
headers2 = ['Model', 'Logic', 'Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC']

def create_table(filename):
    with open(f"results/{filename}.json", "r") as f:  # Load results stored in the .json file
        results = [json.loads(line) for line in f if line.strip()]  # Due to the fact that the file contains a json object per line, instead of a single json
    print(results[0].keys())

    table1 = []
    table2 = []
    for result in results:
        table1.append(
            [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['TP'], result['TN'], result['FP'], result['FN'], result['FPR'],
             result['FNR']])
        table2.append(
            [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['Accuracy'], result['Specificity'], result['Precision'],
             result['Recall'], result['F1Score'],
             result['MCC']])
    table1 = tabulate(table1, headers1, tablefmt="latex_booktabs")
    table2 = tabulate(table2, headers2, tablefmt="latex_booktabs")
    return table1, table2

def main():
    dataset = "XSTest"
    table1, table2 = create_table(f"{dataset}_score_results")
    print(f"\n\t*** Dataset: {dataset} - TABLE 1:***\n")
    print(table1)
    print(f"\n\t*** Dataset: {dataset} - TABLE 2: ***\n")
    print(table2)

    '''
    headers = ["Uno", "Due", "Tre"]
    table = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
    print("\t***\n")
    print(tabulate(table, headers, tablefmt="latex_booktabs"))
    '''

    return

main()
