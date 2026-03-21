# Predicting Adverse Interaction Signals in Mental-Health Polypharmacy Using Machine Learning

A machine learning pipeline for detecting adverse drug-drug interaction (DDI) signals in mental-health polypharmacy using the **TwoSIDES** dataset (derived from FDA FAERS). Built as a final project for HADM 283 - AI in Healthcare at Hofstra University.

**Authors:** Desray K. Desir & TaraJee Clarke

---

## Project Overview

Mental-health polypharmacy — the concurrent use of multiple psychiatric medications — is common in clinical practice and significantly increases the risk of adverse drug-drug interactions (DDIs). Many interaction risks only surface after medications are widely prescribed in real-world settings, highlighting the importance of post-marketing pharmacovigilance.

This project uses **Proportional Reporting Ratio (PRR)** values from TwoSIDES to define interaction signals, then trains machine learning classifiers to identify and prioritize mental-health drug pairs associated with elevated reporting signals.

> **Clinical framing:** This system is a *signal prioritization tool* — not a definitive clinical decision-maker. Its goal is to surface drug combinations that deserve additional scrutiny.

---

## What is PRR?

PRR compares how often a specific adverse event appears with a drug pair versus how often it appears with all other drugs. A PRR > 1 suggests the event is reported more often than expected, indicating a potential safety signal.

```
PRR = [A / (A + B)] / [C / (C + D)]

Where:
  A = reports with both drugs + the event
  B = reports with both drugs + NOT the event
  C = reports with other drugs + the event
  D = reports with other drugs + NOT the event
```

---

## Dataset

| Source | Description |
|--------|-------------|
| **TwoSIDES** | Drug-drug-adverse event associations from FAERS. Each row contains a drug pair, adverse event, contingency counts (A/B/C/D), PRR, and related measures. |

Download from: https://tatonettilab.org/resources/nsides/

> **Note:** Data files are excluded via `.gitignore`. Download TwoSIDES and place it in the project root as `twosides.csv`.

### Mental-Health Drug Scope
Rows were filtered to pairs where at least one drug matched a curated list of:
- SSRIs / SNRIs (fluoxetine, sertraline, escitalopram, venlafaxine, etc.)
- Antipsychotics (quetiapine, olanzapine, risperidone, etc.)
- Benzodiazepines (alprazolam, diazepam, clonazepam, etc.)
- Mood stabilizers (lithium, valproate, lamotrigine, etc.)
- Stimulants (methylphenidate, amphetamine, lisdexamfetamine, etc.)

This yielded ~9,657 rows from a 100,000-row chunk.

---

## Project Structure

```
mental-health-ddi-prediction/
├── 01_data_engineering.py              # Chunked loading, Polars filtering, label creation
├── 02_modeling.py                      # EDA, model training, evaluation, error analysis
├── mental_health_ddi_prediction.ipynb  # Combined end-to-end Jupyter notebook
├── requirements.txt
├── .gitignore
└── README.md
```

**Two-notebook design** (reflecting real scaling constraints):
1. `01_data_engineering` — handles the large TwoSIDES file safely via chunking + Polars
2. `02_modeling` — loads the prepared subset and focuses entirely on ML

---

## Methods

### Data Engineering
| Step | Approach | Reason |
|------|----------|--------|
| Load chunk | `pd.read_csv(nrows=100_000)` | Avoids RAM crashes in Colab |
| Save intermediate CSV | pandas → plain CSV | Avoids Polars encoding errors on raw `.xz` |
| Load into Polars | `encoding="utf8-lossy"`, `ignore_errors=True` | Handles irregular/malformed rows |
| Mental-health filter | Keyword match on drug names | Clinical focus + manageable size |
| Label creation | PRR > 1 → label 1 | Converts continuous signal to supervised format |

### Modeling

All models use the same pipeline structure:
```
OneHotEncoder(handle_unknown="ignore") → Classifier
```

| Model | Configuration |
|-------|--------------|
| LR Baseline | Default logistic regression |
| LR Balanced | `class_weight="balanced"` |
| Random Forest Balanced | 100 trees, `class_weight="balanced"` |

Train/test split: **80% / 20%**, stratified, `random_state=42`

---

## Results

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| LR Baseline | 0.934 | 0.690 |
| LR Balanced | 0.612 | 0.691 |
| Random Forest Balanced | 0.612 | 0.691 |

### Key Findings

**Why baseline accuracy is misleadingly high:** The dataset is heavily imbalanced — TwoSIDES is signal-enriched, so PRR > 1 dominates. A model that predicts label 1 for nearly every case achieves ~93% accuracy while failing completely on the minority class.

**Why balanced models matter clinically:** In pharmacovigilance, missing a true signal (false negative) can delay investigation of a harmful drug combination. Balanced weighting shifts focus toward detecting the minority class, reducing false negatives at the cost of overall accuracy.

**Why Random Forest ≈ LR here:** One-hot encoded drug names are sparse and identity-based. There's no nonlinear structure for the forest to exploit. Richer features (drug class, mechanism, metabolic pathway) would likely close this gap.

---

## Limitations

- Analysis run on a 100K-row chunk of TwoSIDES, not the full corpus
- Features limited to drug names only — no pharmacologic structure encoded
- PRR thresholding discards signal magnitude (continuous → binary simplification)
- One-hot encoding creates sparse, high-dimensional features with no notion of drug similarity
- Class imbalance persists despite weighting; accuracy alone remains a misleading metric

---

## Future Work

- Integrate **DrugCentral** for drug class and mechanism of action features
- Use **SHAP** for feature-level interpretability
- Try **XGBoost / Gradient Boosting** with richer numeric features
- Extend scaling beyond chunking (distributed preprocessing)
- Prototype a CDSS-style interface for PRR-based risk scoring

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/tarajeeclarke/mental-health-ddi-prediction.git
cd mental-health-ddi-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download TwoSIDES
Download from https://tatonettilab.org/resources/nsides/ and place as `twosides.csv` in the project root.

### 4. Run the pipeline

```bash
# Step 1 — Data engineering (generates twosides_mental_health_subset.csv)
python 01_data_engineering.py

# Step 2 — Modeling
python 02_modeling.py
```

Or open the combined notebook:
```bash
jupyter notebook mental_health_ddi_prediction.ipynb
```

---

## Technologies

- Python 3.10+
- pandas, numpy, polars
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

---

## References

- TwoSIDES / NSIDES dataset — Tatonetti Lab: https://tatonettilab.org/resources/nsides/
- FDA Adverse Event Reporting System (FAERS): https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers
- scikit-learn documentation: https://scikit-learn.org
- Polars documentation: https://docs.pola.rs

---

## Authors

**TaraJee Clarke** & **Desray K. Desir**  
MS Health Informatics — Hofstra University  
[LinkedIn](https://www.linkedin.com/in/tarajeeclarke) · [GitHub](https://github.com/tarajeeclarke)
