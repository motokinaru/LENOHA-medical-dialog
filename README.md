
# LENOHA Medical Dialogue System

**Local, zero‑hallucination medical dialogue support for pre‑procedure communication.**

This repository provides a **minimal, one‑command reproducibility package** for the **classifier evaluation** reported in our manuscript (under review). The goal is to let reviewers and readers **reproduce the classification metrics locally** in minutes without any cloud dependency beyond model downloads.

> ℹ️ **Scope**: This package evaluates a **binary classifier** that separates *clinical questions* from *small talk / administrative messages* using FAQ‑based retrieval. **Generation (e.g., small‑talk synthesis with Swallow‑8B)** is *out of scope* here and not included.

---

## Quick Start

### Windows (PowerShell)

```powershell
cd artifact/minimal
powershell -ExecutionPolicy Bypass -File .\run_eval.ps1
```

This script installs dependencies and runs the evaluator against:

* **FAQ**: `artifact/minimal/faq_en.xlsx`
* **Test set**: repository‑root `./test_en.csv`
* **Output folder**: `artifact/minimal/out/`

### Linux / macOS

```bash
cd artifact/minimal
python -m pip install -r requirements.txt
python ./tch_eval_classify.py \
  --faq_xlsx ./faq_en.xlsx \
  --input_csv ../../test_en.csv \
  --st_model intfloat/multilingual-e5-large-instruct \
  --threshold 0.905 \
  --out_dir ./out \
  --seed 42 --log_env
```

**Outputs** will appear in `artifact/minimal/out/`:

* `metrics.json`, `metrics.csv` — Accuracy / Precision / Recall / F1 / Specificity / **Balanced Accuracy** + **95% Wilson CI**, **AUC**, **AUPRC**
* `preds.csv` — per‑row score and predicted label
* `misclassified.csv` — FP / FN rows
* `roc_points.csv`, `pr_points.csv` — curve point dumps for plotting
* (if `--log_env`) `env.json`, `requirements.lock`

---

## What’s Included (minimal)

```
artifact/minimal/
  requirements.txt
  run_eval.ps1
  tch_eval_classify.py     # evaluation script (thresholded classifier + metrics + ROC/AUC/AUPRC)
  faq_en.xlsx              # sample FAQ (header: question, answer)
# repo root
  test_en.csv              # sample evaluation data (columns: input,label)
```

### Input formats

* **FAQ (`faq_en.xlsx`)**: Column **A** = `question`, Column **B** = `answer` (row 1 is header).
* **Test CSV (`test_en.csv`)**: Columns `input,label` where `label` is **1 = clinical question**, **0 = small talk / admin**.

> The provided `test_en.csv` contains **200 lines** (101 small‑talk/admin, 100 clinical) crafted for reproducibility; it is **free of PHI** and suitable for public release.

---

## Evaluator details (`tch_eval_classify.py`)

* **Model**: `SentenceTransformer` embeddings (default `intfloat/multilingual-e5-large-instruct`).

  * The **E5 prompt convention** is applied internally: `query: ...` / `passage: ...`.
* **Scoring**: For each input, compute cosine similarity (on normalized embeddings) to all FAQ questions; use **max similarity** as the score.
* **Decision rule**: Threshold at **`--threshold 0.905`** (default). The paper selected this by **Youden index on validation**; feel free to override for sensitivity analyses.
* **Metrics**:

  * Confusion matrix (TN, FP, FN, TP)
  * Accuracy, Precision, Recall, F1, Specificity, **Balanced Accuracy**
  * **95% Wilson confidence intervals** for Accuracy / Precision / Recall / Specificity
  * **ROC/AUC** and **PR/AUPRC** (curve points exported)
* **Reproducibility switches**:

  * `--seed 42` (sets NumPy/Random and best‑effort Torch seeds; exact determinism of embedding inference is not required)
  * `--log_env` (writes `env.json` and a `requirements.lock` via `pip freeze`)
* **Comparison**: `--compare_preds path/to/other_model_preds.csv` (must contain a `pred` column) adds **McNemar’s test** (with continuity correction) and **Cohen’s h**.

---

## Customizing / Using Your Own Data

1. Prepare your FAQ as **Excel** with columns: `question, answer` (row 1 = header).
2. Prepare your test set as **CSV** with columns: `input,label` (1 = clinical, 0 = small talk/admin).
3. Run the evaluator, pointing to your files:

   ```bash
   python ./tch_eval_classify.py \
     --faq_xlsx /path/to/your_faq.xlsx \
     --input_csv /path/to/your_test.csv \
     --st_model intfloat/multilingual-e5-large-instruct \
     --threshold 0.905 \
     --out_dir ./out_custom
   ```

### Tips

* **GPU** is optional; CPU works (GPU just accelerates encoding).
* First run will download the model from Hugging Face; ensure internet is available.
* If using a virtual environment, point PowerShell script’s `$py` to your venv’s `python.exe`.

---

## Rationale & Design

* We target **local, retrieval‑anchored classification** to reduce hallucination risk in pre‑procedure communication flows.
* The minimal package purposely **does not require any clinic data**. The shipped examples are **toy / synthetic** and demonstrate the pipeline end‑to‑end.
* We **encourage sharing** of scripts, thresholds, and any artifacts (e.g., ROC/PR CSVs) to support **reproducibility** and secondary analyses.

---

## Data Availability

The included `faq_en.xlsx` and `test_en.csv` are **toy samples for reproducibility**. Clinically curated datasets used in our study are **not publicly released** due to ethics and institutional policies. For qualified collaborations, please contact the corresponding author subject to approvals.

---

## License

Released under the **MIT License** (see [`LICENSE`](./LICENSE)).

## Citation

If you use this code, please cite this repository. (Formal bibliographic details will be added after acceptance.)

## Disclaimer

This software is for **research purposes only** and must **not** be used for clinical decision‑making, diagnosis, or treatment.

---

## (Optional) Japanese Quick Guide / 日本語クイックガイド

**実行方法（Windows）**

```powershell
cd artifact/minimal
powershell -ExecutionPolicy Bypass -File .\run_eval.ps1
```

**実行方法（Linux / macOS）**

```bash
cd artifact/minimal
python -m pip install -r requirements.txt
python ./tch_eval_classify.py \
  --faq_xlsx ./faq_en.xlsx \
  --input_csv ../../test_en.csv \
  --st_model intfloat/multilingual-e5-large-instruct \
  --threshold 0.905 \
  --out_dir ./out \
  --seed 42 --log_env
```

出力は `artifact/minimal/out/` に保存されます（`metrics.json/csv`, `preds.csv`, `misclassified.csv`, `roc_points.csv`, `pr_points.csv` など）。
