import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import shap
import os

os.makedirs('models', exist_ok=True)

# ============================================================
# Load cleaned data saved by prepare_data.py
# ============================================================
print("==================== LOADING CLEANED DATA ====================")

X_train_ohe   = pd.read_csv('data/X_train_ohe.csv')
X_test_ohe    = pd.read_csv('data/X_test_ohe.csv')
X_train_le    = pd.read_csv('data/X_train_le.csv')
X_test_le     = pd.read_csv('data/X_test_le.csv')
X_train_smote = pd.read_csv('data/X_train_smote.csv')

y_train       = pd.read_csv('data/y_train.csv').squeeze()
y_test        = pd.read_csv('data/y_test.csv').squeeze()
y_train_smote = pd.read_csv('data/y_train_smote.csv').squeeze()

print(f"[✓] Data Loaded successfully.")
print(f"[✓] Training with SMOTE: {X_train_smote.shape} rows")
print(f"[✓] Testing with Real Data: {X_test_ohe.shape} rows")

# ============================================================
# Step 1: Train Baseline Random Forest Models
# ============================================================
print("\n==================== STEP 1: Train Models ====================")

rf_settings = {
    'n_estimators': 300,
    'max_depth': 30,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_jobs': -1,
    'random_state': 42
}

print("[...] Training Model A — OHE, no SMOTE...")
rf_ohe = RandomForestClassifier(**rf_settings)
rf_ohe.fit(X_train_ohe, y_train)

print("[...] Training Model B — Label Encoding, no SMOTE...")
rf_le = RandomForestClassifier(**rf_settings)
rf_le.fit(X_train_le, y_train)

print("[...] Training Model C — OHE + SMOTE...")
rf_smote = RandomForestClassifier(**rf_settings)
rf_smote.fit(X_train_smote, y_train_smote)

print("[✓] All models trained with identical settings.")

# 1. Evaluate the OHE Model (The Baseline)
print("\n" + "="*20 + " RF OHE REPORT " + "="*20)
y_pred_ohe = rf_ohe.predict(X_test_ohe)
print(classification_report(y_test, y_pred_ohe))

# 2. Evaluate the LE Model (The Comparison)
print("\n" + "="*20 + " RF LE REPORT " + "="*20)
y_pred_le = rf_le.predict(X_test_le)
print(classification_report(y_test, y_pred_le))

# 3. Evaluate the SMOTE Model (The Champion)
print("\n" + "="*20 + " RF SMOTE REPORT " + "="*20)
y_pred_smote = rf_smote.predict(X_test_ohe) # We're using X_test_ohe because SMOTE was built on OHE features
print(classification_report(y_test, y_pred_smote))

# ============================================================
# Step 2: Cross-Validation (5-Fold)
# ============================================================
print("\n==================== STEP 2: Cross-Validation ====================")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("[...] Running 5-fold CV on Model A (OHE)...")
cv_scores_ohe = cross_val_score(
    rf_ohe, X_train_ohe, y_train,
    cv=cv, scoring='f1_macro', n_jobs=-1
)

print("[...] Running 5-fold CV on Model B (Label Encoding)...")
cv_scores_le = cross_val_score(
    rf_le, X_train_le, y_train,
    cv=cv, scoring='f1_macro', n_jobs=-1
)

print("[...] Running 5-fold CV on Model C (OHE + SMOTE)...")
sampling_strategy = {'Analysis': 12000, 'Backdoor': 12000, 'Worms': 12000, 'Shellcode': 12000}
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)),
    ('clf',   RandomForestClassifier(**rf_settings))
])
cv_scores_smote = cross_val_score(
    smote_pipeline, X_train_ohe, y_train,
    cv=cv, scoring='f1_macro', n_jobs=-1
)

print(f"\n[✓] Cross-Validation Results (F1-macro, 5 folds):")
print(f"  Model A — OHE (no SMOTE):   {cv_scores_ohe.mean():.4f}  (+/- {cv_scores_ohe.std():.4f})")
print(f"  Model B — Label Encoding:   {cv_scores_le.mean():.4f}  (+/- {cv_scores_le.std():.4f})")
print(f"  Model C — OHE + SMOTE:      {cv_scores_smote.mean():.4f}  (+/- {cv_scores_smote.std():.4f})")

# Find the winner
scores = {
    'Model A — OHE':          cv_scores_ohe.mean(),
    'Model B — Label Encoding': cv_scores_le.mean(),
    'Model C — OHE + SMOTE':  cv_scores_smote.mean()
}
best_model_name = max(scores, key=scores.get)
if best_model_name == 'Model C — OHE + SMOTE':
    best_model = rf_smote
    best_X_test = X_test_ohe # SMOTE uses OHE features
elif best_model_name == 'Model A — OHE':
    best_model = rf_ohe
    best_X_test = X_test_ohe
else:
    best_model = rf_le
    best_X_test = X_test_le

joblib.dump(best_model, 'models/rf_model_final.pkl')

print(f"\n[✓] Best model by CV score: {best_model_name} ({scores[best_model_name]:.4f})")
print(f"[✓] Winner exported as 'rf_model_final.pkl'")

# ============================================================
# Step 3: Feature Selection
# ============================================================
print("\n==================== STEP 3: Feature Selection ====================")

# --- Method A: Pearson Correlation ---
print("[Method A] Pearson Correlation...")

corr_matrix = X_train_ohe.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_cols = [
    col for col in upper_triangle.columns
    if any(upper_triangle[col] > 0.95)
]

print(f"[✓] Features with correlation > 0.95: {len(high_corr_cols)}")
print(high_corr_cols)

# --- Method B: Random Forest Feature Importance ---
print(f"\n[Method B] Random Forest Feature Importance (Using {best_model_name})...")

importances = pd.Series(
    best_model.feature_importances_,
    index=best_X_test.columns
).sort_values(ascending=False)

print(f"\n[✓] Top 15 most important features:")
print(importances.head(15))

print(f"\n[✓] Bottom 15 least important features:")
print(importances.tail(15))

low_importance = importances[importances < 0.001].index.tolist()
print(f"\n[✓] Features with importance < 0.1% ({len(low_importance)} features):")
print(low_importance)

print("\n[Info] These features are candidates for removal in the next iteration")
print("[Info] Retrain the model without them and compare F1 scores")

# --------Save trained models---------
joblib.dump(rf_ohe,   'models/rf_ohe.pkl')
joblib.dump(rf_le,    'models/rf_le.pkl')
joblib.dump(rf_smote, 'models/rf_smote.pkl')

print("[✓] Models saved → models/")

# ============================================================
# Step 4: Final Evaluation on Test Set
# ============================================================
print("\n==================== STEP 4: Final Evaluation ====================")

for name, model, X_te in [
    ("Model A — OHE (no SMOTE)",  rf_ohe,   X_test_ohe),
    ("Model B — Label Encoding",  rf_le,    X_test_le),
    ("Model C — OHE + SMOTE",     rf_smote, X_test_ohe),
]:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    y_pred = model.predict(X_te)

    print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# Step 5: XAI — SHAP Explainability
# ============================================================
print("\n==================== STEP 5: SHAP Explainability ====================")

print(f"[...] Building SHAP explainer on {best_model_name}...")
explainer = shap.TreeExplainer(best_model)

X_sample = X_test_ohe.iloc[:200]

print("[...] Computing SHAP values...")
shap_values = explainer.shap_values(X_sample)

joblib.dump(explainer, 'models/shap_explainer.pkl')
np.save('models/shap_values_sample.npy', shap_values)

print("[✓] SHAP explainer saved → models/shap_explainer.pkl")
print("[✓] SHAP values saved   → models/shap_values_sample.npy")

print("\n==================== PIPELINE COMPLETE ====================")
print(f"[✓] Best model by CV:  {best_model_name}")