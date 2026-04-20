# Cross-Domain Opinion Mining on Multi-Source Reviews

A text mining and NLP project comparing traditional machine learning and transformer-based approaches for sentiment classification of Amazon product reviews, with cross-domain evaluation on magazine subscription reviews.

 **Course:** Text Mining Project Work, Team 4
 
 **Program:** MSc Data Science & Business Analytics
 
 **Institution:** Bologna Business School, Alma Mater Studiorum — Università di Bologna
 
 **Supervisors:** Prof. Gianluca Moro & Dr. Giacomo Frisoni — DISI, University of Bologna



---

##  Project Overview

This project tackles **cross-domain opinion mining**: training sentiment classifiers on one domain (Amazon Gift Card reviews) and evaluating how well they generalise to a different, unseen domain (Magazine Subscription reviews). Three models of increasing complexity are compared — from a traditional TF-IDF baseline to a fully fine-tuned transformer model.

The core research question: *Can a fine-tuned BERT model generalize sentiment analysis across product domains where vocabulary, style, and context differ significantly?*

---

##  Business Problem

Sentiment analysis systems trained on one domain (e.g., electronics reviews) often fail when applied to another domain (e.g., insurance feedback, financial reports). This is a critical limitation for enterprise NLP pipelines. This project explores whether fine-tuned transformers offer a viable path to domain-agnostic opinion mining.

---

## Data Sources

Two Amazon product review datasets are used:

| Dataset | File | Sample Size | Description |
|---|---|---|---|
| **Reviews A** (source domain) | `Gift_Cards.json.gz` | 100,000 reviews | Amazon Gift Card reviews, rated 1–5 stars |
| **Reviews B** (target domain) | `Magazine_Subscriptions.json.gz` | 50,000 reviews | Amazon Magazine Subscription reviews, rated 1–5 stars |

Both files are downloaded automatically at runtime from Dropbox. Each review contains a `reviewText` field and an `overall` star rating (1–5).

---

##  Methodology

### 1. Data Understanding & Exploration
Both datasets were explored to understand their structure, size, and label distributions. Both datasets are heavily imbalanced — in Reviews A, over 80% of ratings are 5-star; in Reviews B, over 60% are 5-star.

### 2. Class Balancing (Reviews A)
Reviews A was balanced using **undersampling** — reducing all classes to the size of the smallest class — to ensure the model is not biased towards the dominant 5-star rating.

### 3. Preprocessing
- Reviews rated 3 stars (neutral) were removed from the dataset
- Remaining ratings were re-labelled as binary sentiment: `pos` (4–5 stars) and `neg` (1–2 stars)
- Rows with missing `reviewText` values were dropped before vectorisation
- Subword tokenization via BERT WordPiece tokenizer for transformer-based models

### 4. Train / Validation / Test Split
Reviews A was split using an **80-10-10** stratified split:
- **Training set (80%)** — used to fit the model
- **Validation set (10%)** — used for hyperparameter tuning and model selection
- **Test set (10%)** — held out for final evaluation

Stratification ensures a consistent proportion of `pos` and `neg` labels across all three splits.

---

##  Models

Three models were trained and evaluated:

### Model 1 — TF-IDF + Logistic Regression
A classical NLP baseline using a bag-of-words representation:
- **Vectorisation:** TF-IDF with `min_df=5` (removes rare words appearing in fewer than 5 documents) and `ngram_range=(2,2)` (bigrams only, to capture phrase-level context such as "not good")
- **Classifier:** Logistic Regression with L2 regularisation
- **Tuning:** GridSearchCV with 5-fold cross-validation over regularisation parameter C ∈ {0.01, 0.1, 1, 10, 100}
- **Best configuration:** C=100, penalty=L2

### Model 2 — Frozen ModernBERT + Logistic Regression
A feature extraction approach using a pre-trained transformer:
- **Model:** `answerdotai/ModernBERT-base` (~150M parameters), frozen (weights not updated)
- **Embeddings:** The `[CLS]` token representation from the final hidden layer is extracted for each review, producing a fixed-length dense vector
- **Classifier:** Logistic Regression trained on top of these embeddings (C=1.0)
- The base model is not fine-tuned — it is used purely as a feature extractor, testing whether pre-trained representations transfer without adaptation

### Model 3 — Fine-tuned ModernBERT
Full end-to-end fine-tuning of ModernBERT for binary sequence classification:
- **Base model:** `answerdotai/ModernBERT-base`
- **Training:** 1 epoch, learning rate 2e-5, weight decay 0.01, batch size 8
- **Optimiser:** AdamW with linear learning rate warmup scheduler
- **Framework:** HuggingFace `Trainer` API with `TrainingArguments`
- **Labels:** Integer-encoded (`neg=0`, `pos=1`)
- Fine-tuning updates all model weights, allowing the transformer to adapt fully to the sentiment classification task

---

##  Results

### Reviews A (In-Domain — Amazon Gift Cards)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 81.8% | 78.4% | 87.9% | 82.9% |
| Frozen ModernBERT + LR | 83.9% | 82.1% | 86.5% | 84.2% |
| **Fine-Tuned ModernBERT**  | **92.2%** | **91.2%** | **93.3%** | **92.2%** |

### Reviews B (Cross-Domain — Magazine Subscriptions)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 70.5% | 70.4% | 70.7% | 70.6% |
| Frozen ModernBERT + LR | 77.5% | 76.8% | 78.7% | 77.7% |
| **Fine-Tuned ModernBERT**  | **90.0%** | **88.7%** | **91.6%** | **90.1%** |

### Key Findings
- **Fine-Tuned ModernBERT significantly outperforms both baselines** across both in-domain and cross-domain evaluation, demonstrating strong generalisation
- **Cross-domain transfer is hardest for TF-IDF** — performance drops ~11 percentage points from Reviews A to Reviews B, reflecting the model's dependence on domain-specific vocabulary and its inability to handle vocabulary mismatch
- **Frozen BERT improves cross-domain robustness** over TF-IDF, as contextual embeddings capture more transferable semantic features without any task-specific adaptation
- **Fine-tuned ModernBERT shows minimal domain degradation** — only ~2 percentage points from Reviews A to Reviews B — confirming that fine-tuning produces robust, transferable sentiment representations rather than domain-specific surface patterns
- BERT's attention mechanism captures semantic sentiment signals that transfer across domains where vocabulary, style, and context differ significantly

---

## Experiment Tracking

All training and inference runs are logged to **Weights & Biases** under the project `cross-domain-opinion-mining`. Each model run is tracked with:
- Training, validation, and test accuracy
- Training & validation loss curves (for fine-tuned model)
- Accuracy, Precision, Recall, F1 per epoch
- Hyperparameter configurations (learning rate, batch size, warmup steps, max sequence length)
- Run names clearly labelled per experiment (`tfidf-logreg`, `frozen-modernbert-logreg`, `finetuned-modernbert-1epoch`)
- **WandB Sweeps** used for hyperparameter optimisation — optimal config identified: LR=2e-5, batch_size=8, warmup_ratio=0.1

🔗 WandB Project: https://wandb.ai/sukhdeepsinghsaini-bologna-business-school/cross-domain-opinion-mining

---

## Project Structure

```
├── WanDB_project_work_team_Finalversion_submission.ipynb   # Main notebook
├── Gift_Cards.json.gz                                       # Reviews A (auto-downloaded)
└── Magazine_Subscriptions.json.gz                          # Reviews B (auto-downloaded)
```

---

## Running the Project

1. Open `WanDB_project_work_team_Finalversion_submission.ipynb` in **Google Colab** (recommended for GPU access)
2. Enable GPU acceleration: Runtime → Change Runtime Type → GPU
3. Run all cells sequentially — the notebook is structured in 10 tasks:

| Task | Description |
|---|---|
| 1 | Load datasets (Reviews A and Reviews B) |
| 2 | Explore and understand both datasets |
| 3 | Balance classes in Reviews A via undersampling |
| 4 | Preprocess Reviews A (label binarisation, column selection) |
| 5 | Train / validation / test split (80-10-10, stratified) |
| 6 | TF-IDF + Logistic Regression — train, tune, and evaluate on A |
| 7 | Frozen ModernBERT + Logistic Regression — extract embeddings and evaluate on A |
| 8 | Fine-tune ModernBERT for 1 epoch and evaluate on A |
| 9 | Cross-domain evaluation — test all three models on Reviews B |
| 10 | Conclusions and results analysis |

> **Note:** ModernBERT inference on ~90K reviews is GPU-intensive. Ensure GPU is enabled and use a batch size > 1. Clear GPU memory between training and inference phases using `torch.cuda.empty_cache()`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | TF-IDF vectorisation, Logistic Regression, GridSearchCV, metrics |
| PyTorch | Deep learning framework for ModernBERT |
| HuggingFace Transformers | ModernBERT model, tokeniser, Trainer API |
| HuggingFace Datasets | Dataset formatting for fine-tuning |
| NLTK | Text preprocessing (classical pipeline) |
| matplotlib / seaborn | Visualisation (label distributions, confusion matrices) |
| WandB | Experiment tracking and hyperparameter sweeps |
| Google Colab | GPU-accelerated runtime |
