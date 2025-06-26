import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# ── 1. Load aggregated test set ───────────────────────────────────────────────
test_agg = pd.read_csv('test_agg_features.csv')
X_test   = test_agg.drop(['subject', 'label'], axis=1)
y_test   = test_agg['label'].values

# ── 2. Load PCA and transform ─────────────────────────────────────────────────
pca = joblib.load('pca.pkl')
X_test_pca = pca.transform(X_test)
print("Transformed test set to PCA space:", X_test_pca.shape)

# ── 3. Locate and load the best patient‐level model ───────────────────────────
model_files = glob.glob('*_patient_model.pkl')
if not model_files:
    raise FileNotFoundError("No *_patient_model.pkl found. Run aggregate_and_tune.py first.")
model_path = model_files[0]
print("Loading model:", model_path)
model = joblib.load(model_path)

# ── 4. Base evaluation ────────────────────────────────────────────────────────
y_proba = model.predict_proba(X_test_pca)[:, 1]
base_auc = roc_auc_score(y_test, y_proba)
print(f"Base Test ROC AUC: {base_auc:.3f}")

# ── 5. Threshold tuning (Youden's J) ─────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thresh = thresholds[best_idx]
print(f"Optimal threshold by Youden's J: {best_thresh:.3f}")

y_pred_thresh = (y_proba >= best_thresh).astype(int)
cm_thresh = confusion_matrix(y_test, y_pred_thresh)
print("\nConfusion Matrix at threshold:")
print(cm_thresh)
print("\nClassification Report at threshold:")
print(classification_report(y_test, y_pred_thresh))

# ── 6. Bootstrap AUC confidence interval ──────────────────────────────────────
n_bootstraps = 1000
rng = np.random.RandomState(42)
bootstrapped_scores = []

for i in range(n_bootstraps):
    # sample with replacement
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:
        # skip this sample
        continue
    score = roc_auc_score(y_test[indices], y_proba[indices])
    bootstrapped_scores.append(score)

bootstrapped_scores = np.array(bootstrapped_scores)
ci_lower = np.percentile(bootstrapped_scores, 2.5)
ci_upper = np.percentile(bootstrapped_scores, 97.5)
print(f"\nAUC {base_auc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

# ── 7. Save ROC curve with threshold annotation ───────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC={base_auc:.3f})')
plt.scatter(fpr[best_idx], tpr[best_idx], color='red',
            label=f'Youden\'s J threshold\n({best_thresh:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Optimal Threshold')
plt.legend(loc='lower right')
plt.savefig('roc_curve_with_threshold.png', bbox_inches='tight')
plt.close()
print("Saved ROC curve with threshold to roc_curve_with_threshold.png")

