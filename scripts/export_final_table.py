import os
import json
import glob
import numpy as np
import pandas as pd

def export_extensive_csv():
    # Scan for all evaluation test report JSONs
    report_files = glob.glob('runs/**/report_*.json', recursive=True)
    if not report_files:
        report_files = glob.glob('report_*_test.json')

    results = []

    # Base paper published baselines for comparison
    qgdr_rows = [
        {
            'Model / Configuration': 'QGDR Base Paper (EyeQ Subset)', 
            'Accuracy (%)': 78.32, 'Balanced Acc (%)': '-', 'QWK': 0.6863, 
            'Macro-F1': '-', 'Sensitivity (%)': '-', 'Specificity (%)': '-', 
            'AUC-OVR': '-', 'Referable AUC': '-', 'ECE (%)': '-'
        },
        {
            'Model / Configuration': 'QGDR Base Paper (DDR)', 
            'Accuracy (%)': 80.85, 'Balanced Acc (%)': '-', 'QWK': 0.8231, 
            'Macro-F1': '-', 'Sensitivity (%)': '-', 'Specificity (%)': '-', 
            'AUC-OVR': '-', 'Referable AUC': '-', 'ECE (%)': '-'
        }
    ]

    for path in report_files:
        try:
            with open(path, 'r') as f:
                rep = json.load(f)
            
            filename = os.path.basename(path)
            run_name = filename.replace('report_', '').replace('_test.json', '')
            
            ov = rep.get('overall', {})
            cm = np.array(ov.get('confusion_matrix', [[0]*5]*5))
            
            total_samples = np.sum(cm)
            accuracy = np.sum(np.diag(cm)) / total_samples if total_samples > 0 else 0
            
            # Referable DR binary classification calculation (Grade >= 2 vs < 2)
            true_neg = np.sum(cm[0:2, 0:2])
            false_pos = np.sum(cm[0:2, 2:5])
            false_neg = np.sum(cm[2:5, 0:2])
            true_pos = np.sum(cm[2:5, 2:5])
            
            sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
            
            results.append({
                'Model / Configuration': f"Ours: {run_name}",
                'Accuracy (%)': round(accuracy * 100, 2),
                'Balanced Acc (%)': round(balanced_acc * 100, 2),
                'QWK': round(ov.get('qwk', 0), 4),
                'Macro-F1': round(ov.get('macro_f1', 0), 4),
                'Sensitivity (%)': round(sensitivity * 100, 2),
                'Specificity (%)': round(specificity * 100, 2),
                'AUC-OVR': round(ov.get('auc_ovr', 0), 4),
                'Referable AUC': round(ov.get('referable_auc', 0), 4),
                'ECE (%)': round(ov.get('ece', 0) * 100, 2)
            })
        except Exception as e:
            print(f"Error reading {path}: {e}")

    df_ours = pd.DataFrame(results)
    if not df_ours.empty:
        # Sort rows by QWK descending
        df_ours = df_ours.sort_values(by='QWK', ascending=False).reset_index(drop=True)
        df_final = pd.DataFrame(qgdr_rows)._append(df_ours, ignore_index=True)
        
        # Save to reports/ folder
        os.makedirs('reports', exist_ok=True)
        out_path = 'reports/final_comparison_table.csv'
        df_final.to_csv(out_path, index=False)
        
        print("\n=== EXTENSIVE BASE VS OURS TABLE GENERATED SUCCESSFULLY ===")
        print(df_final.to_string(index=False))
        print(f"\nSaved CSV file at: {out_path}")
    else:
        print("No report JSON files found.")

if __name__ == '__main__':
    export_extensive_csv()