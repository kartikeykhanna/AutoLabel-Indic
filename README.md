# AutoLabel-Indic 🇮🇳

### Bootstrapping Intent Classification for Low-Resource Indian Languages (Marathi)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Pipeline](#project-pipeline)
- [Datasets](#datasets)
- [Models Explored](#models-explored)
- [Results](#results)
- [Pseudo-Labeling: Marathi](#pseudo-labeling-marathi)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Team](#team)

---

## Overview

**AutoLabel-Indic** is a data-centric NLP project that addresses the challenge of intent classification for **Marathi** — a low-resource Indian language with over 83 million speakers but virtually no publicly available labeled intent datasets.

Instead of focusing solely on building a classifier, this project emphasizes **automatically generating labeled data** using:

- Self-supervised learning and pre-trained multilingual models
- Cross-lingual transfer from English (SNIPS) to Marathi
- Machine translation (Helsinki-NLP Marian MT: `opus-mt-mr-en`)
- Confidence-filtered **pseudo-labeling** to bootstrap labeled Marathi data

---

## Problem Statement

| Challenge | Description |
|-----------|-------------|
| ❌ No labeled dataset | No large-scale public Marathi intent dataset exists |
| 💸 High annotation cost | Manual labeling requires native speakers and domain expertise |
| 🔄 Language variation | Marathi has dialectal, colloquial, and informal variations |
| ⚠️ Translation noise | Direct translation from English introduces semantic drift |

---

## Project Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  English SNIPS  │────▶│  Train ML Models │────▶│  Marathi Dataset │
│  13,084 samples │     │  (10 classifiers)│     │  (48 sentences)  │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
                                               ┌──────────────────────┐
                                               │  Marian MT           │
                                               │  (mr → en translate) │
                                               └──────────┬───────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │  Predict + Filter    │
                                               │  (confidence ≥ 0.5)  │
                                               └──────────┬───────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │  12 Auto-labeled     │
                                               │  Marathi Samples     │
                                               └──────────────────────┘
```

---

## Datasets

### English: SNIPS (Source)

| Property | Value |
|----------|-------|
| Source | `DeepPavlov/snips` (HuggingFace) |
| Train samples | 13,084 |
| Test samples | 1,400 |
| Intent classes | 7 |

**Intent Labels:**

| ID | Intent |
|----|--------|
| 0 | AddToPlaylist |
| 1 | BookRestaurant |
| 2 | GetWeather |
| 3 | PlayMusic |
| 4 | RateBook |
| 5 | SearchCreativeWork |
| 6 | SearchScreeningEvent |

### Marathi: Custom Dataset (Own)

| Property | Value |
|----------|-------|
| Total sentences | 48 |
| Domain | Utility / Civic service queries |
| Script | Devanagari |
| Labels | None (unlabeled — pseudo-labeled by our pipeline) |
| File | `marathi_unlabeled_dataset.csv` |

**Sample sentences:**

```
माझं बिल चुकीचं आहे          → My bill is incorrect
सेवा नीट चालत नाही           → Service is not working properly
इंटरनेट कनेक्शन बंद आहे      → Internet connection is down
मला नवीन गॅस कनेक्शन हवं आहे → I need a new gas connection
नेटवर्क सतत कट होत आहे       → Network keeps dropping
```

---

## Models Explored

### Classical ML (TF-IDF features)

| Model | Lab/Assessment |
|-------|----------------|
| Logistic Regression | Assess 3 |
| Naive Bayes | Assess 3 |
| Support Vector Machine (SVM) | Assess 3 |
| Multi-Layer Perceptron (MLP) | Batch 12 |
| Decision Tree | Batch 12 |
| Random Forest | Lab 2 |
| AdaBoost | Lab 2 |

### Deep Learning

| Model | Features | Lab |
|-------|----------|-----|
| LSTM | Random embeddings | Lab 4 |
| GRU | Random embeddings | Lab 4 |
| LSTM | GloVe 100d | Assess 5 |
| GRU | GloVe 100d | Assess 5 |

### Transformer

| Model | Lab |
|-------|-----|
| DistilBERT (`distilbert-base-uncased`) | Lab 6, Lab 7 |

---

## Results

### English SNIPS — Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 96.7% | 96.8% | 96.7% | 96.7% |
| Naive Bayes | 95.5% | 95.5% | 95.5% | 95.5% |
| SVM | 97.0% | 97.0% | 97.0% | 97.0% |
| MLP | 97.9% | 97.9% | 97.9% | 97.9% |
| Decision Tree | 92.6% | 92.6% | 92.6% | 92.6% |
| Random Forest | 97.7% | 97.7% | 97.7% | 97.7% |
| AdaBoost | 89.6% | 91.7% | 89.6% | 90.1% |
| LSTM (plain) | 27.1% | 8.2% | 27.1% | 12.4% |
| LSTM + GloVe | 95.9% | 96.0% | 95.9% | 95.9% |
| GRU (plain) | 14.3% | 2.0% | 14.3% | 3.6% |
| **DistilBERT** ⭐ | **99.0%** | **99.0%** | **99.0%** | **99.0%** |

> ⭐ Best overall model: **DistilBERT** fine-tuned for 3 epochs on SNIPS

---

## Pseudo-Labeling: Marathi

The core contribution of this project is the automatic labeling of Marathi text without any manual annotation.

**Pipeline:**

1. Train Random Forest on English SNIPS using TF-IDF features
2. Load Marathi sentences from `marathi_unlabeled_dataset.csv`
3. Translate Marathi → English using `Helsinki-NLP/opus-mt-mr-en` (Marian MT)
4. Vectorize translated text with the same TF-IDF vectorizer
5. Predict intent labels + confidence scores
6. Filter predictions with confidence ≥ 0.5
7. Save high-confidence samples to `marathi_labeled_output.csv`

**Output (12 auto-labeled samples, confidence ≥ 0.5):**

| Marathi Text | Predicted Intent | Confidence |
|--------------|-----------------|------------|
| इंटरनेट कनेक्शन बंद आहे | SearchScreeningEvent | 0.705 |
| माझी तक्रार नोंदवली नाही | SearchScreeningEvent | 0.742 |
| बिल भरण्याची शेवटची तारीख काय आहे | SearchScreeningEvent | 0.608 |
| नवीन कनेक्शनसाठी अर्ज कसा करायचा | SearchScreeningEvent | 0.603 |
| मीटर बदलून द्या | SearchScreeningEvent | 0.566 |

> **Note:** TF-IDF without translation yielded 0 high-confidence samples (confidence stuck at 76.8% due to script mismatch). Translation was the key enabler.

---

## Repository Structure

```
autolabel-indic/
│
├── data/
│   ├── marathi_unlabeled_dataset.csv    # Our custom Marathi dataset (48 sentences)
│   └── marathi_labeled_output.csv       # Auto-labeled output from pseudo-labeling
│
├── notebooks/
│   ├── lab4_slp_lstm_gru.ipynb          # LSTM & GRU (plain embeddings) - Lab 4
│   ├── lab6_distilbert.ipynb            # DistilBERT fine-tuning - Lab 6
│   ├── lab7_distilbert_v2.ipynb         # DistilBERT improved config - Lab 7
│   ├── batch12_classical_ml.ipynb       # LR, NB, SVM on SNIPS - Batch 12
│   ├── batch12_mlp_dt.ipynb             # MLP, DT, RF, AdaBoost - Batch 12
│   ├── assess3_language_detection.ipynb # Language detection experiment - Assess 3
│   ├── assess5_glove_lstm_gru.ipynb     # LSTM & GRU with GloVe - Assess 5
│   └── pseudolabel_marathi.ipynb        # Full pseudo-labeling pipeline
│
├── README.md
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip
- Google Colab (recommended) or local GPU environment

### Install dependencies

```bash
pip install transformers datasets evaluate scikit-learn
pip install tensorflow keras
pip install pandas numpy matplotlib seaborn
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
transformers>=4.0.0
datasets>=2.0.0
evaluate>=0.4.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## How to Run

### 1. Classical ML (LR / NB / SVM)

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

snips = load_dataset("DeepPavlov/snips")
X_train = [s for s in snips["train"]["utterance"]]
y_train = snips["train"]["label"]
X_test  = [s for s in snips["test"]["utterance"]]
y_test  = snips["test"]["label"]

vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = SVC(kernel="linear", probability=True)
model.fit(X_train_vec, y_train)
print("Accuracy:", model.score(X_test_vec, y_test))
```

### 2. DistilBERT Fine-tuning

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

dataset = load_dataset("DeepPavlov/snips")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["utterance"], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(pd.DataFrame(dataset["train"])).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(pd.DataFrame(dataset["test"])).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
trainer.train()
trainer.evaluate()
```

### 3. LSTM with GloVe

```python
# Download GloVe embeddings first
# !wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

EMBEDDING_DIM = 100
MAX_WORDS = 20000
MAX_LEN = 50

lstm_model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, weights=[embedding_matrix],
              input_length=MAX_LEN, trainable=False),
    LSTM(128),
    Dropout(0.5),
    Dense(7, activation="softmax")
])
lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_train_pad, y_train_cat, validation_data=(X_test_pad, y_test_cat), epochs=10, batch_size=64)
```

### 4. Pseudo-Labeling Marathi

```python
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd, numpy as np

# Load Marathi data
marathi_df = pd.read_csv("data/marathi_unlabeled_dataset.csv")

# Translate mr → en
model_name = "Helsinki-NLP/opus-mt-mr-en"
tokenizer  = MarianTokenizer.from_pretrained(model_name)
mt_model   = MarianMTModel.from_pretrained(model_name)

def translate(texts):
    tokens = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in mt_model.generate(**tokens)]

marathi_df["text_en"] = translate(marathi_df["text"])

# Vectorize and predict
marathi_vec  = vectorizer.transform(marathi_df["text_en"])
probs        = rf.predict_proba(marathi_vec)
pred_labels  = rf.classes_[np.argmax(probs, axis=1)]
confidences  = np.max(probs, axis=1)

marathi_df["pseudo_intent"] = pred_labels
marathi_df["confidence"]    = confidences

# Filter high-confidence samples
filtered_df = marathi_df[marathi_df["confidence"] >= 0.5]
filtered_df.to_csv("data/marathi_labeled_output.csv", index=False)
print(f"Auto-labeled {len(filtered_df)} / {len(marathi_df)} samples")
```

---

## Key Findings

**1. Transformers dominate** — DistilBERT at 99% accuracy confirms that pre-trained language models are the gold standard for intent classification, even on relatively small datasets.

**2. GloVe embeddings are essential for RNNs** — LSTM without pre-trained embeddings achieved only 27% accuracy. With GloVe 100d, this jumped to 95.9%, proving that random initialization is insufficient for small training sets.

**3. Classical ML is surprisingly strong** — SVM (97%), MLP (97.9%), and Random Forest (97.7%) are highly competitive baselines and train in seconds vs. hours for transformers.

**4. TF-IDF fails cross-lingually** — An English TF-IDF vectorizer cannot extract meaningful features from Devanagari script. Without translation, all 48 Marathi samples got stuck at the same ~76.8% confidence (model uncertainty). Translation was the key enabler.

**5. GRU collapsed without embeddings** — Plain GRU converged to predicting a single class (accuracy = 14.3% = class frequency). This is a vanishing gradient / initialization issue resolved by GloVe.

**6. 12 auto-labeled Marathi samples achieved** — With confidence threshold 0.5 and Marian MT translation, the Random Forest pseudo-labeler successfully assigned intent labels to 25% of the Marathi dataset — entirely without human annotation.

---

## Future Work

- [ ] Expand Marathi dataset to 500+ sentences covering all 7 intent classes
- [ ] Fine-tune `ai4bharat/indic-bert` or `xlm-roberta-base` directly on pseudo-labeled Marathi data
- [ ] Iterative pseudo-labeling: use labeled samples to retrain and re-label remaining data
- [ ] Extend pipeline to Hindi, Tamil, Telugu, and Bengali
- [ ] Build a Marathi intent classification API for civic/utility chatbots

