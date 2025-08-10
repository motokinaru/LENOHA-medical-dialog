LENOHA Medical Dialogue System
Local, zero-hallucination medical dialogue support for pre-procedure communication.
This repository provides a minimal, one-command reproducibility package for the classifier evaluation reported in our manuscript (under review).

Quick Repro
Windows (PowerShell)

Open PowerShell in this folder: artifact/minimal

Run:
cd artifact/minimal
powershell -ExecutionPolicy Bypass -File .\run_eval.ps1

Outputs will appear in artifact/minimal/out/:

metrics.json, metrics.csv (Accuracy / Precision / Recall / F1 / Specificity + 95% Wilson CI, AUC)

preds.csv (scores and predicted labels)

misclassified.csv (FP/FN)

roc_points.csv

Using a venv? Edit the first line of run_eval.ps1 and set:
$py = "C:/path/to/your/venv/Scripts/python.exe"

Linux / macOS
cd artifact/minimal
python -m pip install -r requirements.txt
python ./tch_eval_classify.py
--faq_xlsx ./faq.xlsx
--input_csv ./test.csv
--st_model intfloat/multilingual-e5-large-instruct
--threshold 0.905
--out_dir ./out

What’s Included (minimal)
artifact/minimal/

README.md

requirements.txt

run_eval.ps1

tch_eval_classify.py (evaluation script: thresholded classifier + metrics + ROC/AUC)

faq.xlsx (sample FAQ: header = question, answer)

test.csv (sample evaluation data: columns = input,label)

Input Formats
faq.xlsx: column A = question, column B = answer (row 1 is a header).

test.csv: columns input,label (1 = clinical question; 0 = small talk).

Outputs
Confusion matrix (tn, fp, fn, tp), Accuracy / Precision / Recall / F1 / Specificity / Balanced Accuracy,
95% Wilson confidence intervals, and AUC.

Misclassified examples (misclassified.csv).

ROC curve points (roc_points.csv).

Optional McNemar / Cohen’s h:
Add --compare_preds path/to/other_model_preds.csv (must contain a pred column).

Models
SentenceTransformer (embeddings): intfloat/multilingual-e5-large-instruct
(E5 prompt convention is applied internally: “query:” / “passage:”)

Threshold: 0.905 (picked via Youden index on validation in the study).

Note: The manuscript’s small-talk generation with Swallow-8B is a separate module.
This minimal package focuses on classification evaluation (zero-generation via FAQ).

Environment
Python 3.11+ recommended.

Pinned dependencies are listed in artifact/minimal/requirements.txt.

GPU is optional (CPU works; GPU accelerates embedding).

Data Availability
The included faq.xlsx and test.csv are toy samples.
Clinically curated datasets used in the study are not publicly released due to ethics and institutional policies. For qualified research collaborations, contact the corresponding author subject to institutional approvals.

License
MIT License (see the repository root LICENSE).

Citation
Please cite this repository if you use the code.
(Formal bibliographic details will be added after acceptance.)

Disclaimer
This software is for research purposes only and must not be used for clinical decision-making, diagnosis, or treatment.
