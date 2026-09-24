import os
import json
import glob
import numpy as np
import pandas as pd

def generate_full_clinical_report():
    report_files = glob.glob('runs/**/report_*.json', recursive=True)
    if not report_files:
        report_files = glob.glob('report_*_test.json')

    results = []

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
            
            # Referable DR binary breakdown (Grade >= 2 vs < 2)
            true_neg = np.sum(cm[0:2, 0:2])
            false_pos = np.sum(cm[0:2, 2:5])
            false_neg = np.sum(cm[2:5, 0:2])
            true_pos = np.sum(cm[2:5, 2:5])
            
            sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
            
            ppv = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            npv = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0
            youden_j = sensitivity + specificity - 1
            
            # F2-Score (emphasizing recall)
            beta = 2
            f2_score = (1 + beta**2) * (ppv * sensitivity) / ((beta**2 * ppv) + sensitivity) if (ppv + sensitivity) > 0 else 0

            results.append({
                'Model / Run': run_name,
                'Accuracy (%)': round(accuracy * 100, 2),
                'Balanced Acc (%)': round(balanced_acc * 100, 2),
                'QWK': round(ov.get('qwk', 0), 4),
                'Macro-F1': round(ov.get('macro_f1', 0), 4),
                'Sensitivity (%)': round(sensitivity * 100, 2),
                'Specificity (%)': round(specificity * 100, 2),
                'PPV / Precision (%)': round(ppv * 100, 2),
                'NPV (%)': round(npv * 100, 2),
                'Youden Index (J)': round(youden_j, 4),
                'F2-Score': round(f2_score, 4),
                'AUC-OVR': round(ov.get('auc_ovr', 0), 4),
                'Referable AUC': round(ov.get('referable_auc', 0), 4),
                'ECE (%)': round(ov.get('ece', 0) * 100, 2)
            })
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='QWK', ascending=False).reset_index(drop=True)
        print("\n=== COMPLETE CLINICAL & ADVANCED METRICS TABLE ===")
        print(df.to_string(index=False))
        
        os.makedirs('reports', exist_ok=True)
        out_path = 'reports/complete_advanced_metrics.csv'
        df.to_csv(out_path, index=False)
        print(f"\nSaved successfully to {out_path}")

if __name__ == '__main__':
    generate_full_clinical_report()