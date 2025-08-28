Quick Repro

Windows (PowerShell)
cd artifact/minimal
powershell -ExecutionPolicy Bypass -File .\run_eval.ps1

Linux / macOS
cd artifact/minimal
./run_eval.sh

Outputs (artifact/minimal/out/):
- metrics.json, metrics.csv  # Accuracy / Precision / Recall / F1 / Specificity / Balanced Acc + 95% CI, AUC, AUPRC
- preds.csv                  # scores and predicted labels
- misclassified.csv          # FP/FN rows
- roc_points.csv, pr_points.csv
