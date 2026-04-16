# AI-Powered Intrusion Detection System (IDS)

A machine-learning-based Network Intrusion Detection System trained on the **UNSW-NB15** dataset. The system classifies network flows into 10 categories (Normal + 9 attack types) using Random Forest models and exposes results through an interactive **Streamlit** dashboard with live packet capture, PCAP file analysis, model comparison, and SHAP explainability.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard Sections](#dashboard-sections)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Example Outputs](#example-outputs)
- [Challenges and Limitations](#challenges-and-limitations)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This project implements an end-to-end IDS pipeline:

1. **Data preparation** — cleans, encodes, scales, and oversamples the UNSW-NB15 network traffic dataset.
2. **Model training** — trains and evaluates three Random Forest variants, selects the best by cross-validation, and computes SHAP explainability values.
3. **Interactive dashboard** — a Streamlit web app for real-time monitoring, offline PCAP analysis, model comparison, and AI explainability.

### Attack Classes Detected

| Class | Description |
|---|---|
| Normal | Benign network traffic |
| Generic | Generic attack patterns |
| Exploits | Exploitation of vulnerabilities |
| Fuzzers | Fuzzing-based attacks |
| DoS | Denial of Service |
| Reconnaissance | Network scanning / probing |
| Analysis | Packet analysis-based attacks |
| Backdoor | Backdoor connections |
| Shellcode | Shellcode injection |
| Worms | Self-replicating worm traffic |

---

## System Architecture

```
UNSW-NB15 CSV Files
        │
        ▼
┌──────────────────┐
│  prepare_data.py  │  ← clean, encode (OHE / LE), scale, SMOTE
└────────┬─────────┘
         │  saves CSV files + scalers/encoders to models/
         ▼
┌──────────────────┐
│  train_model.py   │  ← train 3 RF models, cross-validate, SHAP
└────────┬─────────┘
         │  saves .pkl model files to models/
         ▼
┌──────────────────┐
│     app.py        │  ← Streamlit dashboard
│                   │    (live capture, PCAP, comparison, XAI)
└──────────────────┘
```

---

## Project Structure

```
PFE test/
├── data/
│   ├── UNSW_NB15_training-set.csv    ← raw training data (input)
│   ├── UNSW_NB15_testing-set.csv     ← raw testing data (input)
│   ├── X_train_ohe.csv               ← OHE-encoded training features
│   ├── X_test_ohe.csv                ← OHE-encoded test features
│   ├── X_train_le.csv                ← label-encoded training features
│   ├── X_test_le.csv                 ← label-encoded test features
│   ├── X_train_smote.csv             ← SMOTE-oversampled training features
│   ├── y_train.csv                   ← training labels
│   ├── y_test.csv                    ← test labels
│   └── y_train_smote.csv             ← SMOTE-oversampled training labels
│
├── models/
│   ├── rf_ohe.pkl                    ← Model A: OHE, no SMOTE
│   ├── rf_le.pkl                     ← Model B: Label Encoding, no SMOTE
│   ├── rf_smote.pkl                  ← Model C: OHE + SMOTE
│   ├── rf_model_final.pkl            ← best model by CV score
│   ├── scaler_ohe.pkl                ← MinMaxScaler for OHE pipeline
│   ├── scaler_le.pkl                 ← MinMaxScaler for LE pipeline
│   ├── le_proto.pkl                  ← LabelEncoder for protocol
│   ├── le_service.pkl                ← LabelEncoder for service
│   ├── le_state.pkl                  ← LabelEncoder for connection state
│   ├── shap_explainer.pkl            ← SHAP TreeExplainer
│   └── shap_values_sample.npy        ← pre-computed SHAP values (200 samples)
│
├── src/
│   ├── prepare_data.py               ← data cleaning & preprocessing pipeline
│   ├── train_model.py                ← model training, evaluation & SHAP
│   └── app.py                        ← Streamlit dashboard (22 steps)
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- **Windows only**: install [Npcap](https://npcap.com/) to enable live packet capture
- On Windows, run the terminal as **Administrator** when using the Live Detection section

### 1. Clone the repository

```bash
git clone <repository-url>
cd "PFE test"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the two UNSW-NB15 CSV files inside the `data/` directory:

```
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

> The dataset is available from the [UNSW Canberra Cyber research page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

---

## Usage

Run the three scripts **in order** from the project root directory.

### Step 1 — Prepare the data

```bash
python src/prepare_data.py
```

Cleans the raw CSVs, applies encoding and scaling, runs SMOTE, and writes all artefacts to `data/` and `models/`.

### Step 2 — Train the models

```bash
python src/train_model.py
```

Trains three Random Forest models, runs 5-fold stratified cross-validation, selects the best model, computes SHAP values, and saves all `.pkl` files to `models/`.

### Step 3 — Launch the dashboard

```bash
streamlit run src/app.py
```

Opens the dashboard in your default browser at `http://localhost:8501`.

> On Windows, open the terminal as **Administrator** before running the above command to allow live packet capture.

---

## Dashboard Sections

| Section | Description |
|---|---|
| **Dataset Overview** | Class distributions, feature type breakdown, Pearson correlation heatmap, per-feature histograms and box plots |
| **Live Detection** | Real-time Scapy packet capture → bidirectional flow extraction → per-flow classification with confidence scores and colour-coded alerts |
| **PCAP Analysis** | Upload a `.pcap` / `.pcapng` file for offline batch classification with downloadable CSV results |
| **Model Comparison** | Accuracy / Precision / Recall / F1 side-by-side comparison, per-class F1 bar chart, and interactive confusion matrix |
| **Explainability** | Global SHAP feature importance (mean |SHAP| across all classes) and per-sample local explanations with directional colour coding |

### Live Detection — Sidebar Controls

| Control | Description |
|---|---|
| Interface | Network interface to sniff (or auto-detect) |
| Model | Which of the three RF models to use |
| Attack Alert Threshold | Minimum model confidence (20–80 %, default 40 %) to raise an attack alert. Normal traffic is trusted whenever it is the argmax class. |

---

## Technologies Used

| Category | Library / Tool | Version |
|---|---|---|
| Machine Learning | scikit-learn | >= 1.3 |
| Class Imbalance | imbalanced-learn | >= 0.11 |
| Data Processing | pandas, numpy, scipy | >= 2.0 / 1.24 / 1.10 |
| Explainability | shap | >= 0.44 |
| Visualisation | Plotly, Matplotlib, Seaborn | >= 5.18 / 3.7 / 0.13 |
| Dashboard | Streamlit | >= 1.28 |
| Packet Capture | Scapy | >= 2.5 |
| Serialisation | joblib | >= 1.3 |

---

## Dataset

**UNSW-NB15** — created by the Australian Centre for Cyber Security (ACCS), UNSW Canberra.

| Split | Total Samples | Normal | Attack |
|---|---|---|---|
| Training | 175,341 | 56,000 | 119,341 |
| Testing | 82,332 | 37,000 | 45,332 |

**42 raw features** per flow: 39 numerical + 3 categorical (`proto`, `service`, `state`).

#### Preprocessing Summary

| Step | Description |
|---|---|
| Missing values | `service` NaN → `"none"` |
| Rare grouping | Proto values < 1,000 occurrences → `"other"`; state values < 15 → `"other"` |
| Outlier capping | Winsorization at 99th percentile for 15 high-variance columns |
| Encoding A | One-Hot Encoding → 65 features (Models A & C) |
| Encoding B | Label Encoding → 42 features (Model B) |
| Scaling | MinMaxScaler fitted on training data only, applied to both splits |
| SMOTE | Analysis / Backdoor / Worms / Shellcode oversampled to 12,000 each |

---

## Example Outputs

**Figure 1:** [Insert — Dataset Overview: class distribution bar charts for training and test sets]

**Figure 2:** [Insert — Model Comparison: grouped bar chart of Accuracy / Precision / Recall / F1 for all three models]

**Figure 3:** [Insert — Confusion Matrix for Model C (OHE + SMOTE)]

**Figure 4:** [Insert — SHAP Global Feature Importance: top 25 features ranked by mean |SHAP| value]

**Figure 5:** [Insert — SHAP Local Explanation: per-sample waterfall showing features pushing toward attack vs. normal]

**Figure 6:** [Insert — Live Detection: capture log table with red (attack), yellow (uncertain), and white (normal) rows]

**Figure 7:** [Insert — PCAP Analysis: pie chart of predicted traffic categories from an uploaded file]

---

## Challenges and Limitations

### Domain Shift
The models were trained exclusively on UNSW-NB15 lab traffic from 2015. Live internet traffic from modern networks has different statistical characteristics, resulting in lower model confidence on real-world flows. This is a fundamental limitation of any model trained on a static benchmark dataset.

### Feature Extraction Approximation
Live feature extraction via Scapy approximates the UNSW-NB15 feature set. Application-layer features (`trans_depth`, `response_body_len`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`) cannot be recovered from raw packets and are set to zero, which shifts the feature distribution away from training data.

### Class Imbalance
Several attack classes are heavily underrepresented in the raw dataset (Worms: 130 samples, Shellcode: 1,133, Backdoor: 1,746). SMOTE oversampling mitigates this, but recall on these classes remains lower than for high-frequency classes such as Generic or Exploits.

### Privilege Requirements
Live packet capture requires administrator/root privileges and (on Windows) the Npcap driver.

---

## Future Work

- **Modern datasets** — retrain on CIC-IDS-2018 or CIC-IoT-2023 to reduce domain shift with contemporary traffic.
- **Deep learning** — explore LSTM or Transformer architectures for temporal sequence modelling of network flows.
- **Online / incremental learning** — adapt the model continuously as new labelled flows are observed.
- **Application-layer parsing** — integrate Zeek or Suricata to accurately extract HTTP/FTP/DNS features currently set to zero.
- **Alert enrichment** — add GeoIP lookup, reverse DNS resolution, and threat-intelligence feed integration to attack alerts.
- **SIEM integration** — export alerts to Elasticsearch, Splunk, or any syslog-compatible SIEM.

---

## Contributors

| Name |
| Younes Sahraoui |
| Oussama Zayane |

---

## License

This project is currently unlicensed. All rights reserved by the author(s).  
Contact the repository owner for usage permissions.
