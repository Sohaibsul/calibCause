import os
import json
import glob
import pandas as pd

def generate_results_table():
    report_files = glob.glob('runs/**/report_*.json', recursive=True)
    summaries = []
    
    for path in report_files:
        try:
            with open(path, 'r') as f:
                rep = json.load(f)
            
            ov = rep.get('overall', rep)
            run_name = os.path.basename(path).replace('report_', '').replace('_test.json', '')
            
            summaries.append({
                'Configuration / Run': run_name,
                'Accuracy': round(ov.get('accuracy', 0) * 100, 2),
                'QWK': round(ov.get('qwk', 0), 4),
                'Macro-F1': round(ov.get('macro_f1', 0), 4),
                'AUC-OVR': round(ov.get('auc_ovr', 0), 4),
                'Referable AUC': round(ov.get('referable_auc', 0), 4),
                'ECE (%)': round(ov.get('ece', 0) * 100, 2)
            })
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            
    df = pd.DataFrame(summaries)
    if not df.empty:
        df = df.sort_values(by='QWK', ascending=False).reset_index(drop=True)
        print("\n=== FINAL RESULTS SUMMARY TABLE ===")
        print(df.to_string(index=False))
        
        # Save to CSV
        os.makedirs('reports', exist_ok=True)
        df.to_csv('reports/results_summary_vscode.csv', index=False)
        print("\nSaved table successfully to: reports/results_summary_vscode.csv")
    else:
        print("No evaluation reports found under runs/ directory.")

if __name__ == '__main__':
    generate_results_table()