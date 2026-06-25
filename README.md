# 🧠 Natural Language Processing (NLP) Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Transformers%20%7C%20NLTK%20%7C%20spaCy-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> A comprehensive NLP pipeline covering text preprocessing, classification, named entity recognition, sentiment analysis, and more — built with modern Python NLP libraries.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This project provides an end-to-end NLP pipeline that handles various natural language understanding tasks. From raw text ingestion to model inference, the pipeline is modular and extensible, making it easy to plug in new datasets or swap out models.

**Key use cases:**
- Text Classification (multi-class / binary)
- Sentiment Analysis
- Named Entity Recognition (NER)
- Text Summarization
- Question Answering
- Language Modeling

---

## ✨ Features

- ✅ Modular text preprocessing (tokenization, lemmatization, stopword removal)
- ✅ Classical ML baselines (TF-IDF + Logistic Regression, Naive Bayes)
- ✅ Deep learning models (LSTM, CNN for text)
- ✅ Transformer-based models (BERT, DistilBERT via HuggingFace)
- ✅ Named Entity Recognition with spaCy
- ✅ Sentiment analysis pipeline
- ✅ Evaluation metrics: Accuracy, F1, Precision, Recall, BLEU, ROUGE
- ✅ Experiment tracking with MLflow
- ✅ Configurable via YAML config files

---

## 🗂️ Project Structure

```
nlp-project/
│
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Cleaned and tokenized data
│   └── external/               # External embeddings (GloVe, FastText)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Text cleaning and tokenization
│   └── 03_modeling.ipynb       # Model training and evaluation
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py          # Text cleaning utilities
│   │   ├── tokenizer.py        # Tokenization logic
│   │   └── vectorizer.py       # TF-IDF, word embeddings
│   │
│   ├── models/
│   │   ├── baseline.py         # Logistic Regression, Naive Bayes
│   │   ├── lstm_model.py       # LSTM-based classifier
│   │   └── transformer.py      # HuggingFace BERT wrapper
│   │
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy, F1, BLEU, ROUGE
│   │
│   └── utils/
│       ├── config.py           # Config loader
│       └── logger.py           # Logging utilities
│
├── configs/
│   └── config.yaml             # Model and training hyperparameters
│
├── tests/
│   └── test_preprocessing.py   # Unit tests
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip / conda
- (Optional) GPU with CUDA for transformer fine-tuning

### Clone the Repository

```bash
git clone https://github.com/<your-username>/nlp-project.git
cd nlp-project
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🚀 Usage

### 1. Preprocessing

```bash
python src/preprocessing/cleaner.py --input data/raw/text.csv --output data/processed/clean.csv
```

### 2. Train a Baseline Model

```bash
python src/models/baseline.py --config configs/config.yaml
```

### 3. Fine-tune BERT

```bash
python src/models/transformer.py \
  --model bert-base-uncased \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --epochs 3 \
  --batch_size 16
```

### 4. Run Inference

```python
from src.models.transformer import NLPClassifier

model = NLPClassifier.load("outputs/bert_finetuned/")
prediction = model.predict("The product quality exceeded all expectations!")
print(prediction)  # {'label': 'POSITIVE', 'score': 0.98}
```

---

## 🧩 Modules

### Preprocessing

| Function | Description |
|----------|-------------|
| `clean_text()` | Lowercasing, punctuation removal, URL stripping |
| `tokenize()` | Word/sentence tokenization using NLTK or spaCy |
| `lemmatize()` | Reduces words to base form |
| `remove_stopwords()` | Filters common stopwords |
| `vectorize()` | TF-IDF or dense embedding representation |

### Models

| Model | Library | Task |
|-------|---------|------|
| Logistic Regression | scikit-learn | Classification |
| Naive Bayes | scikit-learn | Classification |
| LSTM | PyTorch | Sequence modeling |
| BERT / DistilBERT | HuggingFace Transformers | Fine-tuning |
| spaCy NER | spaCy | Named Entity Recognition |

---

## 📊 Datasets

| Dataset | Task | Size | Source |
|---------|------|------|--------|
| IMDb Reviews | Sentiment Analysis | 50K | [HuggingFace Datasets](https://huggingface.co/datasets/imdb) |
| CoNLL-2003 | NER | 22K sentences | [CoNLL-2003](https://huggingface.co/datasets/conll2003) |
| AG News | Text Classification | 120K | [HuggingFace Datasets](https://huggingface.co/datasets/ag_news) |
| Custom Dataset | Domain-specific | Variable | Place in `data/raw/` |

To use a custom dataset, format it as a CSV with `text` and `label` columns:

```csv
text,label
"Great product, highly recommend.",positive
"Terrible experience, would not buy again.",negative
```

---

## 🤖 Models

### Configuration (`configs/config.yaml`)

```yaml
model:
  name: bert-base-uncased
  num_labels: 2
  max_length: 128

training:
  epochs: 5
  batch_size: 32
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01

data:
  train_path: data/processed/train.csv
  val_path: data/processed/val.csv
  test_path: data/processed/test.csv

logging:
  mlflow_uri: http://localhost:5000
  experiment_name: nlp-bert-classification
```

---

## 📈 Results

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| TF-IDF + Logistic Regression | 87.3% | 0.871 | Baseline |
| LSTM (128 units) | 89.1% | 0.889 | GloVe embeddings |
| DistilBERT (fine-tuned) | 93.6% | 0.934 | 3 epochs |
| BERT-base (fine-tuned) | **94.8%** | **0.947** | 5 epochs |

> Results on IMDb sentiment analysis test set.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure all tests pass and code is formatted with `black`.

```bash
black src/ tests/
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [MLflow](https://mlflow.org/)
