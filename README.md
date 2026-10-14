# NASA IMS Bearing Predictive Maintenance Pipeline

An end-to-end data engineering and prognostic health management (PHM) framework that streams high-frequency vibration datasets to isolate, track, and classify rotating machinery degradation states in real time. 

The architecture processes sequential sensor feeds, extracts rolling time-domain statistical indicators, and evaluates prognosis classification performance using sequential partitioning to eliminate future-lookahead data leakage.

## 📊 System Architecture & Workspace Layout
```text
nasa-bearing-pm/
├── data/
│   ├── raw/            # Extracted IMS run-to-failure folder channels (2nd_test)
│   └── processed/      # Compressed engineered feature data matrices (Parquet)
├── src/
│   ├── __init__.py     # Package initialization directory map
│   ├── utils.py        # Stream validation data loader & test condition specifiers
│   ├── pipeline.py     # High-frequency windowed signal processing pipeline
│   └── train.py        # Sequential evaluation framework & Random Forest classifier
├── main.py             # Root execution manager & CLI engine
└── requirements.txt    # Project platform library dependencies
```

## ⚙️ Feature Engineering Methodology
Every data snapshot represents a 1-second physical vibration record sampled at 20 kHz (20,480 samples). The engineering pipeline streams these records file-by-file to extract critical geometric domain indicators tracking structural breakdown metrics:
* **Root Mean Square (RMS):** Tracks overall running destructive energy growth and structural fatigue acceleration.
* **Kurtosis:** Monitors signal waveform "spikiness" to highlight localized micro-cracking propagation before the metal defects wear flat.
* **Peak-to-Peak:** Captures extreme, instantaneous cyclic physical impact shocks occurring inside the acquisition window.
* **Crest Factor:** Isolates sharp impulse spikes relative to the baseline running energy trend line to catch early structural defects.

## 🔒 Evaluation & Validation Strategy
Because run-to-failure lifetime signatures are highly contiguous time-series tracking data, standard unstratified random shuffling causes severe look-ahead data leakage between neighboring near-identical snapshots. 
* **Chronological Stratification:** Data partition boundaries are split sequentially within each individual health stage phase to ensure the severe breakdown data is represented fairly.
* **Buffer Isolation:** A 5-snapshot buffer gap is enforced directly at the boundaries of the train-test split to eliminate structural edge bleeding.
* **Feature Filtering:** Time progression identifiers are stripped out of the core model training matrix features to eliminate risk of artificial label leakage.

## 📊 Performance Matrix & Verification Results
Running the system orchestrator across the full run-to-failure lifecycle pipeline yields the following production metrics:

```text
      NASA IMS BEARING FAULT DIAGNOSIS LIFESPAN EVALUATION
============================================================
                   precision    recall  f1-score   support

           Normal       0.98      0.73      0.84       172
Early_Degradation       0.56      0.86      0.68        69
     Severe_Fault       0.79      0.95      0.86        40

         accuracy                           0.79       281
        macro avg       0.78      0.85      0.79       281
     weighted avg       0.85      0.79      0.80       281

Macro F1: 0.7939
```

### Top Feature Importances (Physical Grounding Validation)
The mathematical feature rankings match the physical experiment layout completely. `b1_rms` and `b1_std` are isolated as the top primary prediction features because **Bearing 1** is the exact component undergoing failure during this run sequence.

```text
Top 10 features by importance:
b1_rms             0.1501
b1_std             0.1478
b2_std             0.1350
b2_rms             0.1083
b3_std             0.0825
b3_rms             0.0774
```

## 🚀 Local Execution Instructions

1. Install project package dependencies using `uv` or standard virtual environments:
```bash
uv venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

2. Place your raw timestamped dataset files inside `data/raw/2nd_test/` and trigger the execution pipeline:
```bash
uv run python main.py
```
