import os
import pandas as pd
import json
import sys
from tabulate import tabulate, tabulate_formats

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

    # Convert to data frame for further processing (bold best result per category)
    df1 = pd.DataFrame(table1, columns=headers1)
    df2 = pd.DataFrame(table2, columns=headers2)

    # Calculate best values for table 2
    for col in ['Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC']:
        best = df2[col].max()
        worst = df2[col].min()
        df2[col] = df2[col].apply(
            lambda x: "--" if x == None
            else f"\\textbf{{{x * 100:.2f}}}" if x == best
            else f"\\underline{{{x * 100:.2f}}}" if x == worst
            else f"{x * 100:.2f}"
        )

    # Calculate best FPR and best FNR for table 1: Lower is better
    for col in ['FPR', 'FNR']:
        best = df1[col].min()
        worst = df1[col].max()
        df1[col] = df1[col].apply(
            lambda x: "--" if x == None
            else f"\\textbf{{{x * 100:.2f}}}" if x == best
            else f"\\underline{{{x * 100:.2f}}}" if x == worst
            else f"{x * 100:.2f}"
        )

    #table1 = tabulate(df1, headers1, tablefmt="latex_booktabs_raw", showindex=False)
    #table2 = tabulate(df2, headers2, tablefmt="latex_booktabs_raw", showindex=False)

    table1 = tabulate(df1, headers1, tablefmt="latex_raw", showindex=False)
    table2 = tabulate(df2, headers2, tablefmt="latex_raw", showindex=False)
    return table1, table2

def extract_rows(tabulate_output):
    lines = tabulate_output.strip().split("\n")
    return "\n".join(lines[4:-2])

def main():
    #dataset = "XSTest(all_categories)"
    dataset = "MedSafetyBench"
    #dataset = "PatientSafetyBench(all_categories)"
    table1, table2 = create_table(f"{dataset}_score_results")
    table1 = extract_rows(table1)
    table2 = extract_rows(table2)
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

    '''
    # Table template
    
    \begin{table}[h]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular} ... \end{tabular}
    }
    \caption{Your caption here}
    \label{tab:your_label}
    \end{table}
    '''

    """
# Long table template (with small)
{\small
\begin{longtable}{llrrrrrr}
\caption{5 Models benchmark on XSTest} \label{tab:5models} \\
 
\toprule
Model & Logic & TP & TN & FP & FN & FPR & FNR \\
\midrule
\endfirsthead
 
\multicolumn{8}{c}{\tablename\ \thetable{} -- continued from previous page} \\
\toprule
Model & Logic & TP & TN & FP & FN & FPR & FNR \\
\midrule
\endhead
 
\midrule
\multicolumn{8}{r}{Continued on next page} \\
\endfoot
 
\bottomrule
\endlastfoot
        # HERE CONTENT OF TABLE
\end{longtable}
}
    """

    return

main()
