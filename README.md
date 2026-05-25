# Diabetes Risk Predictor

End-to-end machine learning project that predicts diabetes risk from eight
routine health indicators (Pima Indians Diabetes Dataset, 768 patients) and
serves the trained model through a Streamlit dashboard.

## Project Overview

- **Research question.** Can machine learning models accurately predict diabetes risk based on patient health indicators?
- **Models trained.** Logistic Regression (baseline) and Random Forest (final).
- **Final model.** Random Forest — chosen for the best ROC-AUC, recall, and accuracy on the held-out test set.
- **Deployment.** Interactive Streamlit dashboard with sliders for the eight features and a real-time risk readout.


## Dataset

- **Source.** [Pima Indians Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Rows / columns.** 768 / 9 (8 features + outcome)
- **Class balance.** ~65 % non-diabetic, ~35 % diabetic
- **Cohort.** Female patients of Pima Indian heritage aged 21+

| Feature | Description |
| --- | --- |
| Pregnancies | Number of times pregnant |
| Glucose | 2-h OGTT plasma glucose (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-h serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Family-history risk score |
| Age | Age in years |
| Outcome | 1 = diabetic, 0 = non-diabetic |

## Methods

1. **Cleaning.** Replace biologically impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with NaN, then median-impute.
2. **EDA.** Outcome balance, feature histograms, Glucose / BMI boxplots, correlation heatmap.
3. **Split.** 80 / 20 train / test, stratified on outcome.
4. **Scaling.** StandardScaler for Logistic Regression; Random Forest uses raw values.
5. **Models.** Logistic Regression (baseline) and Random Forest (200 trees).
6. **Evaluation.** Accuracy, precision, recall, F1, and ROC-AUC on the held-out test set.
7. **Explainability.** Random Forest feature importance and SHAP summary plot.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |
| **Random Forest** | **0.7403** | **0.6522** | **0.5556** | **0.6000** | **0.8166** |

Random Forest wins on every metric and is the selected model. Glucose, BMI, and Age are the dominant predictors per both impurity-based importance and SHAP attribution.

Plots are saved to [`images/`](images/) and reproduced in [`diabetes_analysis.ipynb`](diabetes_analysis.ipynb).

## Dashboard

Interactive Streamlit app with four sections:

1. **Introduction** — project overview and disclaimer
2. **Patient inputs** — sidebar sliders for all eight features
3. **Prediction** — live risk percentage, label, and threshold visualization
4. **Model insights** — feature importance, SHAP, ROC curves, confusion matrix

Run locally:

```bash
streamlit run streamlit_app.py
```

## Installation

```bash
# 1. Clone or download this directory
# 2. (Optional) Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models (writes models/*.pkl and images/*.png)
python3 train_models.py

# 5. Run the dashboard
streamlit run streamlit_app.py
```

The notebook is already executed with outputs in place — open it directly with Jupyter or VS Code.

## File Layout

```
.
├── Diabetes/
│   └── diabetes.csv              # raw dataset
├── images/                       # all plots (generated)
├── models/                       # trained model + scaler + metrics.json (generated)
├── report/
│   └── final_report.docx         # written report (generated, draft)
├── diabetes_analysis.ipynb       # main notebook, Phases 1–8
├── streamlit_app.py              # dashboard, Phase 9
├── train_models.py               # reproducible training pipeline
├── build_notebook.py             # rebuilds the .ipynb from source
├── build_report.py               # rebuilds the .docx with current metrics
├── requirements.txt
└── README.md
```

## Future Improvements

- **Add XGBoost** — install `libomp` (`brew install libomp` on macOS) and add a `XGBClassifier(eval_metric='logloss')` to `train_models.py`.
- **Hyperparameter tuning** — `GridSearchCV` or `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`.
- **Cross-validation** — replace the single hold-out with stratified k-fold to get a more reliable performance estimate.
- **Threshold tuning** — pick the decision threshold from the precision-recall curve rather than defaulting to 0.5, to push recall higher.
- **Class imbalance** — try `class_weight='balanced'` or SMOTE on the training set.
- **Deployment** — Streamlit Cloud, Render, or HuggingFace Spaces for a public-facing dashboard.
- **External validation** — re-test on a different cohort (e.g. NHANES) to check generalization beyond the Pima Indian population.
