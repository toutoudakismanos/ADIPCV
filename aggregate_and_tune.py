# aggregate_and_tune.py

import warnings
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition      import PCA
from sklearn.linear_model      import LogisticRegression
from xgboost                   import XGBClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import GridSearchCV, StratifiedKFold
from sklearn.metrics           import roc_auc_score, roc_curve, confusion_matrix, classification_report
import joblib

# ── 0. Suppress unwanted warnings ──────────────────────────────────────────────
warnings.filterwarnings('ignore', 'Parameters: .*use_label_encoder.*')
warnings.filterwarnings('ignore', 'Only one class is present in y_true.*')

# ── 1. Load slice‐level feature CSVs ───────────────────────────────────────────
train_df = pd.read_csv('train_radiomics_features.csv')
test_df  = pd.read_csv('test_radiomics_features.csv')

# ── 2. Aggregate per subject (mean & std) ─────────────────────────────────────
def aggregate_by_subject(df):
    grp = df.groupby('subject')
    agg = grp.aggregate(['mean','std'])
    agg.columns = [f"{feat}_{stat}" for feat,stat in agg.columns]
    agg = agg.reset_index()
    labels = grp['label'].first().reset_index()
    return agg.merge(labels, on='subject')

train_agg = aggregate_by_subject(train_df)
test_agg  = aggregate_by_subject(test_df)

X_train = train_agg.drop(['subject','label'], axis=1)
y_train = train_agg['label']
X_test  = test_agg.drop( ['subject','label'], axis=1)
y_test  = test_agg['label']

# ── 3. PCA to capture 95% variance ─────────────────────────────────────────────
pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
print("PCA reduced to", X_train_pca.shape[1], "components (from", X_train.shape[1], "features)")

# ── 4. Define models & hyperparameter grids ───────────────────────────────────
models = {
    'LogReg_L1': (
        LogisticRegression(
            penalty='l1', solver='saga',
            class_weight='balanced',
            max_iter=10000, random_state=42
        ),
        {'C': [0.01, 0.1, 1, 10]}
    ),
    'RandomForest': (
        RandomForestClassifier(class_weight='balanced', random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
    ),
    'XGBoost': (
        XGBClassifier(
            eval_metric='logloss',
            random_state=42
        ),
        {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
    )
}

# ── 5. Tune with StratifiedKFold (no subject leakage) ─────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}

for name, (mdl, params) in models.items():
    print(f"\n=== Tuning {name} ===")
    gs = GridSearchCV(
        mdl, params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    gs.fit(X_train_pca, y_train)
    best_models[name] = gs.best_estimator_
    print(" Best params:", gs.best_params_)
    print(" CV AUC:   ", gs.best_score_)

# ── 6. Plot patient‐level ROC curves ───────────────────────────────────────────
plt.figure(figsize=(6,6))
for name, mdl in best_models.items():
    proba = mdl.predict_proba(X_test_pca)[:,1]
    auc   = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# diagonal
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Patient‐level ROC Curves')
plt.legend(loc='lower right')
plt.savefig('patient_level_roc.png', bbox_inches='tight')
plt.close()
print("Saved patient_level_roc.png")

# ── 7. Save best patient‐level model ──────────────────────────────────────────
best_name, best_model = max(
    best_models.items(),
    key=lambda kv: roc_auc_score(y_test, kv[1].predict_proba(X_test_pca)[:,1])
)
joblib.dump(best_model, 'patient_model.pkl')
print("Saved best model:", best_name)
joblib.dump(pca, 'pca.pkl')
print("Saved PCA to pca.pkl")

train_agg.to_csv('train_agg_features.csv', index=False)
test_agg.to_csv('test_agg_features.csv',  index=False)
print("Saved aggregated features to train_agg_features.csv / test_agg_features.csv")
