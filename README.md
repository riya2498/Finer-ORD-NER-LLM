# 💸 Financial Named Entity Recognition (NER) with BERT

Fine-tuned Transformer model for identifying financial entities like `ORG`, `MONEY`, `DATE`, and `PERSON` in financial texts. Built using Hugging Face 🤗 Transformers and trained on a curated dataset from Kaggle.

> 📈 Ideal for financial news tagging, document analysis, and automated report generation.

---

## 🚀 Project Highlights

- 🔍 **Domain-specific fine-tuning** on the FiNER-ORD dataset
- 🤖 Based on `bert-base-cased` with token-level classification
- 📊 Supports evaluation with F1, precision, recall, and confusion matrix
- ⚙️ Easy-to-run inference pipeline for live predictions

---

## 📁 Project Structure

```bash
.
├── financial_ner_model/         # Saved model directory
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── notebook.ipynb               # Full training and evaluation notebook
├── inference.py                 # Script for running inference
├── requirements.txt
└── README.md

⚙️ Installation & Setup

1. Clone the repo

git clone https://github.com/yourusername/financial-ner.git
cd financial-ner

2. Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

🧠 Training the Model
Run the full pipeline in the notebook:

notebook.ipynb  # Open in Google Colab or Jupyter

Key Steps:

Load and preprocess the FiNER-ORD dataset

Fine-tune bert-base-cased

Run evaluation and error analysis

Visualize metrics

🔍 Run Inference
Option 1: Using Python Script
from inference import run_inference

text = "Tesla shares jumped 7% after Elon Musk's earnings call."
print(run_inference(text))

Option 2: Hugging Face Pipeline (after training)
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("./financial_ner_model")
tokenizer = AutoTokenizer.from_pretrained("./financial_ner_model")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ner_pipeline("Goldman Sachs expects inflation to cool by Q4 2023.")

📊 Example Output
Text: Apple shares rose 5% in Q1 2023
Entities:
- Apple → ORG
- 5% → PERCENT
- Q1 2023 → DATE

📈 Evaluation Results

Experiment	Epochs	F1 Score	Accuracy
Exp 1	3	0.850	98.48%
Exp 2	4	0.877	98.60%
Exp 3	5	0.868	98.57%
🔧 Future Improvements
Switch to FinBERT or SpanBERT for better domain fit

Add UI using Streamlit or Gradio

Deploy with Hugging Face Spaces

🙌 Contributing
Contributions are welcome! Please open an issue or submit a pull request. Ideas for improvement include:

Enhancing the dataset

Improving entity coverage

Optimizing training time

📜 License
MIT License © 2025 riya2498
