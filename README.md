# ReneWind — Wind Turbine Failure Prediction (Deep Learning)

Predicting wind‑turbine generator failures from anonymized sensor data using **Artificial Neural Networks (Keras / TensorFlow)** so that maintenance teams can repair components **before** they fail — drastically reducing replacement costs.

> Built as part of my Deep Learning portfolio. Compares **7 neural‑network configurations** (optimizers, momentum, dropout, batch normalization, weight initialization) and selects the best model based on a **business‑driven cost metric**: **Recall**.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Experiments](#model-experiments)
- [Results](#results)
- [Final Model & Business Impact](#final-model--business-impact)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## Business Problem

**ReneWind** is a renewable‑energy company that runs a fleet of wind turbines. Sensors on every turbine record temperature, vibration, wind speed, gearbox/blade/brake telemetry, etc. The data has been **ciphered** for confidentiality (40 anonymized predictors `V1 … V40`).

The objective is to build a **binary classifier** that predicts generator **failure (1)** vs **no failure (0)** so that turbines can be serviced *predictively* instead of reactively.

The cost structure is asymmetric:

| Outcome | Meaning | Cost |
|---|---|---|
| True Positive (TP) | Failure correctly predicted | Repair cost (low) |
| **False Negative (FN)** | **Real failure missed by model** | **Replacement cost (highest)** |
| False Positive (FP) | Predicted failure that didn't happen | Inspection cost (lowest) |
| True Negative (TN) | Healthy turbine, no alert | $0 |

Because **FNs are the most expensive**, the priority metric is **Recall**, with **F1‑score** as a tie‑breaker. Plain accuracy is misleading due to class imbalance (failures are rare).

---

## Dataset

| | Records | Predictors | Target |
|---|---:|---:|---|
| `Train.csv` | 20,000 | 40 (`V1`–`V40`) | `Target` (0 / 1) |
| `Test.csv`  |  5,000 | 40 (`V1`–`V40`) | `Target` (0 / 1) |

> **The raw CSVs are *not* committed** to this repo (they are project‑provided and confidential). To run the notebook end‑to‑end, place `Train.csv` and `Test.csv` inside the `data/` folder. See [`data/README.md`](data/README.md).

---

## Approach

1. **Data Overview** — shape, dtypes, missing values, duplicates, statistical summary.
2. **Exploratory Data Analysis**
   - Univariate analysis of all 40 `V` features (histograms + boxplots).
   - Class imbalance check on `Target`.
   - Correlation heatmap and bivariate analysis.
3. **Preprocessing**
   - Stratified Train / Validation / Test split (split *before* imputation to avoid leakage).
   - Median imputation for missing values.
   - Feature scaling with `MinMaxScaler`.
4. **Model Building** — sequential Keras MLPs with sigmoid output for binary classification, `binary_crossentropy` loss, monitored on **Recall**, **Precision**, **F1**, **Accuracy**.
5. **Model Comparison** — train / validation curves, evaluation table across 7 configurations.
6. **Final Evaluation** on the held‑out test set + confusion matrices.
7. **Business Insights & Recommendations.**

Reproducibility is fixed via `keras.utils.set_random_seed(812)` and `tf.config.experimental.enable_op_determinism()`.

---

## Model Experiments

Seven feed‑forward neural networks were trained and compared (50 epochs, batch size 64):

| # | Model | Architecture | Optimizer | Regularization |
|---|---|---|---|---|
| 0 | Baseline | `[14, 7]` ReLU | SGD | — |
| 1 | SGD + Momentum | `[14, 7]` ReLU | SGD (momentum=0.9) | — |
| 2 | Adam | `[14, 7]` ReLU | Adam | — |
| 3 | Adam + Dropout | `[28, 14]` ReLU | Adam | Dropout(0.3) |
| 4 | **Adam + BatchNorm** | `[28, 14]` ReLU | Adam | BatchNorm |
| 5 | Dropout + BatchNorm | `[28, 14]` ReLU | Adam | Dropout + BatchNorm |
| 6 | He Init + Dropout | `[28, 14]` ReLU (he_normal) | Adam | Dropout(0.3) |

---

## Results

### Validation Performance

| Model | Val Recall | Val Precision | Val F1 | Val Accuracy |
|---|---:|---:|---:|---:|
| 0 — Baseline (SGD)              | 0.33 | 0.82 | 0.47 | 0.959 |
| 1 — SGD + Momentum              | 0.52 | 0.99 | 0.69 | 0.973 |
| 2 — Adam                        | 0.83 | 0.98 | 0.90 | 0.989 |
| 3 — Adam + Dropout              | 0.74 | 0.97 | 0.84 | 0.984 |
| **4 — Adam + BatchNorm**        | **0.91** | **0.94** | **0.92** | **~0.99** |
| 5 — Dropout + BatchNorm         | 0.75 | 0.95 | 0.85 | ~0.98 |
| 6 — He Init + Dropout           | 0.78 | 0.94 | 0.86 | ~0.98 |

### Test‑Set Performance (Unseen Data)

- **Model 4 (BatchNorm)** generalizes the best: **Recall ≈ 0.85**, **Precision ≈ 0.94**, highest F1.
- Model 2 (Adam) is the runner‑up but misses more failures.
- Models 0 and 1 are unsuitable for production because of low recall, despite acceptable accuracy.

---

## Final Model & Business Impact

> **Selected: Model 4 — Adam + Batch Normalization**

- **Detects ~85 %+ of real generator failures on unseen data** → directly avoids the most expensive outcome (full replacement).
- **Maintains ~94 % precision** → false alarms (and unnecessary inspections) are kept low.
- **Highest F1 score** → best overall balance.
- Demonstrates that **Batch Normalization** stabilises training on this anonymized sensor data far better than dropout or vanilla SGD.

**Recommendations**

1. Deploy Model 4 as a daily/near‑real‑time scoring service that flags at‑risk turbines.
2. Pair the alerts with a tiered inspection workflow (alert → quick inspection → repair) so that the small FP cost is bounded.
3. Schedule periodic retraining with newly collected sensor data to handle drift.
4. Continue investing in sensor coverage: model performance directly tracks signal quality.

---

## Repository Structure

```
.
├── README.md                              # You are here
├── LICENSE                                # MIT
├── requirements.txt                       # Pinned Python dependencies
├── .gitignore
├── notebooks/
│   └── ReneWind_Wind_Turbine_Failure_Prediction.ipynb
├── reports/
│   └── ReneWind_Wind_Turbine_Failure_Prediction.html   # Rendered notebook
├── data/
│   └── README.md                          # How to obtain Train.csv / Test.csv
└── images/                                # (Optional) screenshots for README
```

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/varun-dev0/DeepLearning-renewind-wind-turbine-failure-prediction.git
cd DeepLearning-renewind-wind-turbine-failure-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the data

Drop the project‑provided `Train.csv` and `Test.csv` into the `data/` folder.
See [`data/README.md`](data/README.md) for details.

### 5. Run the notebook

```bash
jupyter notebook notebooks/ReneWind_Wind_Turbine_Failure_Prediction.ipynb
```

> Prefer to just *read* the analysis? Open
> [`reports/ReneWind_Wind_Turbine_Failure_Prediction.html`](reports/ReneWind_Wind_Turbine_Failure_Prediction.html)
> in any browser — no setup required.

---

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras 2.19** — model building & training
- **scikit‑learn 1.6** — preprocessing, train/test split, metrics
- **pandas 2.2**, **NumPy 2.0**
- **Matplotlib 3.10**, **Seaborn 0.13** — EDA & visualization
- **Jupyter Notebook**

---

## Future Work

- Hyperparameter search with **Keras Tuner** / **Optuna**.
- Class‑imbalance techniques: **SMOTE**, **class_weight**, focal loss.
- Threshold tuning on `predict_proba` to push recall further.
- Compare with classical ML baselines (XGBoost / Random Forest).
- Wrap the final model in a **FastAPI** microservice + Dockerfile for deployment.
- Add **MLflow** tracking and a CI workflow for reproducible runs.

---

## Author

**Varun Tandon** — [@varun-dev0](https://github.com/varun-dev0)
Portfolio project — Deep Learning Specialization.

If you found this useful, a ⭐ on the repo is appreciated!

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
