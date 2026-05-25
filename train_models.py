"""Train Logistic Regression + Random Forest on Pima diabetes data, save artifacts.

Outputs:
    models/final_model.pkl       — Random Forest (chosen model)
    models/scaler.pkl            — StandardScaler fitted on train features (for LR)
    models/feature_names.pkl     — feature order
    models/feature_defaults.pkl  — median values, for the Streamlit input defaults
    models/metrics.json          — per-model metrics
    images/outcome_distribution.png
    images/feature_histograms.png
    images/glucose_bmi_boxplots.png
    images/correlation_heatmap.png
    images/confusion_matrix_rf.png
    images/roc_curves.png
    images/feature_importance_rf.png
    images/shap_summary.png
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "Diabetes" / "diabetes.csv"
MODELS_DIR = ROOT / "models"
IMG_DIR = ROOT / "images"
MODELS_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# 1. Load
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# 2. Clean — zero values in these columns are physiologically impossible
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
imputer = SimpleImputer(strategy="median")
df[zero_invalid_cols] = imputer.fit_transform(df[zero_invalid_cols])

# 3. EDA plots
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(5, 4))
df["Outcome"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
)
ax.set_xticklabels(["Non-diabetic (0)", "Diabetic (1)"], rotation=0)
ax.set_ylabel("Count")
ax.set_title("Outcome Distribution")
plt.tight_layout()
plt.savefig(IMG_DIR / "outcome_distribution.png", dpi=120)
plt.close()

key_feats = ["Age", "BMI", "Glucose", "Insulin"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flatten(), key_feats):
    ax.hist(df[col], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(col)
plt.suptitle("Distributions of Key Features", y=1.02)
plt.tight_layout()
plt.savefig(IMG_DIR / "feature_histograms.png", dpi=120, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x="Outcome", y="Glucose", ax=axes[0], palette=["#4C72B0", "#DD8452"])
axes[0].set_title("Glucose by Outcome")
sns.boxplot(data=df, x="Outcome", y="BMI", ax=axes[1], palette=["#4C72B0", "#DD8452"])
axes[1].set_title("BMI by Outcome")
plt.tight_layout()
plt.savefig(IMG_DIR / "glucose_bmi_boxplots.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(IMG_DIR / "correlation_heatmap.png", dpi=120)
plt.close()

# 4. Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 5. Scale (for LR)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. Train models
models = {}

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_s, y_train)
models["LogisticRegression"] = (lr, X_test_s)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
models["RandomForest"] = (rf, X_test)

# 7. Evaluate
metrics = {}
for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    metrics[name] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", metrics[name]["roc_auc"])

with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 8. Confusion matrix for the chosen final model (RF)
cm = confusion_matrix(y_test, rf.predict(X_test))
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-diabetic", "Diabetic"],
    yticklabels=["Non-diabetic", "Diabetic"],
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig(IMG_DIR / "confusion_matrix_rf.png", dpi=120)
plt.close()

# 9. ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(lr, X_test_s, y_test, ax=ax, name="Logistic Regression")
RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_title("ROC Curves — Model Comparison")
plt.tight_layout()
plt.savefig(IMG_DIR / "roc_curves.png", dpi=120)
plt.close()

# 10. Feature importance (RF)
importance = rf.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
    "importance", ascending=True
)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp_df["feature"], imp_df["importance"], color="#4C72B0")
ax.set_title("Random Forest — Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(IMG_DIR / "feature_importance_rf.png", dpi=120)
plt.close()

# 11. SHAP summary on RF
# TreeExplainer on RF returns a list (one array per class); take the diabetic class.
explainer = shap.TreeExplainer(rf)
shap_raw = explainer.shap_values(X_test)
if isinstance(shap_raw, list):
    shap_values = shap_raw[1]
elif shap_raw.ndim == 3:
    shap_values = shap_raw[:, :, 1]
else:
    shap_values = shap_raw

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(IMG_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# 12. Persist artifacts
joblib.dump(rf, MODELS_DIR / "final_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
joblib.dump(X.median().to_dict(), MODELS_DIR / "feature_defaults.pkl")

print("\nDone. Metrics:")
print(json.dumps(metrics, indent=2))
