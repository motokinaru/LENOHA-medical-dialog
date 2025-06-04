
# LENOHA Medical Dialogue System

This repository contains the code developed for the study "**Toward Responsible AI in Healthcare: A Local, Zero-Hallucination Medical Dialogue System Aligned with Sustainable Development Goals**" submitted to *npj Digital Medicine*.

The system is designed to be fully executable on a local machine without requiring cloud services, enabling privacy-preserving and sustainable digital health interventions. It consists of:

- **FAQ-based Question Classifier**: Using Sentence Transformer models for efficient patient inquiry classification.
- **Small Talk Generator**: Powered by Swallow-8B-Instruct v0.3 for generating safe and friendly small talk responses.
- **Local Execution**: No internet connection or external API is required, ensuring data privacy.

---

Features

- **Fully Local Execution**: No cloud or external API dependencies.
- **Privacy-Preserving**: No patient data leaves the local environment.
- **Low Resource Requirements**: Runs on consumer-grade laptops.
- **Sustainable**: Minimal energy consumption, supporting SDG 7, 12, and 13.
- **Equity-Oriented**: Suitable for low-resource or rural healthcare settings.

---

Requirements

- Python 3.9+
- torch
- transformers
- sentence-transformers
- pandas
- scikit-learn
- openpyxl
- accelerate

Install all dependencies:

```bash
pip install -r requirements.txt

How to Use　this
1. FAQ-based Question Classifier
Prepare:

ikamerafaq.xlsx: Excel file with FAQ questions and answers.

ikamerazatudan.csv: CSV file with input sentences for classification evaluation.

Run:

bash
コピーする
編集する
python faq_classifier.py
2. Small Talk Generator
Prepare:

Install the Swallow-8B-Instruct v0.3 model from HuggingFace.

Run:

bash
コピーする
編集する
python small_talk_chat.py
Type patient utterances to interact with the chatbot.
Type exit to terminate the session.

Models Used here
Sentence Transformer Models

sonoisa/sentence-bert-base-ja-mean-tokens

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

intfloat/multilingual-e5-large

intfloat/multilingual-e5-large-instruct

Small Language Model

tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3

Output
CSV file containing classification results and confidence scores.

Chat logs for small talk interactions (saved locally).

Record of patient inquiries in Excel format.

License
This project is licensed under the MIT License.

Citation
If you use this code, please cite:

[Citation details will be added after publication]

Disclaimer
This system is intended for research purposes only and is not approved for any clinical decision-making or patient diagnosis.

Note: Due to ethical and privacy considerations, the FAQ data and test datasets used in this project are not publicly available. Researchers interested in accessing the datasets may contact the corresponding author under appropriate agreements.
