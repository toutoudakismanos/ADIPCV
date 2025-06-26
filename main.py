# Updated main.py with SimpleITK-based DICOM I/O to avoid PixelData errors

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from radiomics import featureextractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

# ── 1. Load metadata and labels ────────────────────────────────────────────────
metadata_path = 'D:/Dataset/manifest-1542731172463/metadata.csv'
labels_path   = 'D:/Dataset/manifest-1542731172463/QIN-Breast_TreatmentResponse2014-12-16.xlsx'

metadata = pd.read_csv(metadata_path)
labels_df = pd.read_excel(labels_path)

# ── 2. Rename for merging ──────────────────────────────────────────────────────
metadata = metadata.rename(columns={
    'Data Description URI': 'SubjectID',
    'File Location':        'FileLocation'
})
labels_df = labels_df.rename(columns={
    'Patient ID': 'SubjectID',
    'Response':   'TreatmentResponse'
})

# ── 3. Make absolute paths ────────────────────────────────────────────────────
DICOM_ROOT = r'D:/Dataset/manifest-1542731172463'
metadata['FileLocation'] = (
    metadata['FileLocation']
      .str.replace(r'^[\.\\/]+', '', regex=True)
      .apply(lambda rel: os.path.normpath(os.path.join(DICOM_ROOT, rel)))
)

# ── 4. Binary labels ───────────────────────────────────────────────────────────
labels_df['label'] = labels_df['TreatmentResponse'] \
    .str.lower() \
    .map({'pcr': 1, 'non-pcr': 0})

# ── 5. Merge and check ─────────────────────────────────────────────────────────
merged = metadata.merge(
    labels_df[['SubjectID','label']],
    on='SubjectID',
    how='inner'
)
print(f"Rows after merge: {len(merged)} (must be >0)")

# ── 6. Train/Test split ───────────────────────────────────────────────────────
subjects = merged['SubjectID'].unique()
train_subj, test_subj = train_test_split(
    subjects,
    test_size=0.2,
    stratify=merged.drop_duplicates('SubjectID')['label'],
    random_state=42
)

# ── 7. Dataset class with SimpleITK for robust I/O ────────────────────────────
class RadiomicsDataset(Dataset):
    def __init__(self, df, subj_list, transform=None):
        self.df = df[df['SubjectID'].isin(subj_list)].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        loc = row['FileLocation']

        # If it's a directory, read the full series and take the first slice
        if os.path.isdir(loc):
            reader = sitk.ImageSeriesReader()
            series_files = reader.GetGDCMSeriesFileNames(loc)
            reader.SetFileNames(series_files)
            volume = reader.Execute()
            arr = sitk.GetArrayFromImage(volume)  # shape [slices, H, W]
            img = arr[0].astype(np.float32)
        else:
            # Single-file DICOM
            sitk_img = sitk.ReadImage(loc)
            arr = sitk.GetArrayFromImage(sitk_img)
            if arr.ndim == 3:
                img = arr[0].astype(np.float32)
            else:
                img = arr.astype(np.float32)

        # Generate Otsu mask
        thresh = threshold_otsu(img)
        mask = (img > thresh).astype(np.uint8)

        sample = {
            'image':   img,
            'mask':    mask,
            'label':   row['label'],
            'subject': row['SubjectID']
        }
        return self.transform(sample) if self.transform else sample

# ── 8. Instantiate & sanity-check ─────────────────────────────────────────────
train_ds = RadiomicsDataset(merged, train_subj)
test_ds  = RadiomicsDataset(merged, test_subj)

print("Train slices:", len(train_ds))
print(" Test slices:", len(test_ds))

s = train_ds[0]
print("Keys:",  s.keys())
print("Shape:", s['image'].shape, "Mask vals:", np.unique(s['mask']), "Label:", s['label'])

plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.imshow(s['image'], cmap='gray');    plt.title('Image');    plt.axis('off')
plt.subplot(1,2,2); plt.imshow(s['mask'],  cmap='gray');    plt.title('Otsu mask'); plt.axis('off')
plt.show()

# ── 9. Radiomics feature extraction (train) ───────────────────────────────────
params = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',
    'enableCExtensions': True
}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

train_records = []
for i in range(len(train_ds)):
    sample = train_ds[i]
    sitk_img  = sitk.GetImageFromArray(sample['image'])
    sitk_mask = sitk.GetImageFromArray(sample['mask'])
    feats = extractor.execute(sitk_img, sitk_mask)
    rec = {'subject': sample['subject'], 'label': sample['label']}
    for k, v in feats.items():
        if k.startswith('original_'):
            rec[k] = v
    train_records.append(rec)

train_features_df = pd.DataFrame(train_records)
train_features_df.to_csv('train_radiomics_features.csv', index=False)
print("Saved features to train_radiomics_features.csv")

# ── 10. Radiomics feature extraction (test) ────────────────────────────────────
test_records = []
for i in range(len(test_ds)):
    sample = test_ds[i]
    sitk_img  = sitk.GetImageFromArray(sample['image'])
    sitk_mask = sitk.GetImageFromArray(sample['mask'])
    feats = extractor.execute(sitk_img, sitk_mask)
    rec = {'subject': sample['subject'], 'label': sample['label']}
    for k, v in feats.items():
        if k.startswith('original_'):
            rec[k] = v
    test_records.append(rec)

test_features_df = pd.DataFrame(test_records)
test_features_df.to_csv('test_radiomics_features.csv', index=False)
print("Saved features to test_radiomics_features.csv")

# ── 11. Train classifier with cross-validation ────────────────────────────────
X_train = train_features_df.drop(['subject','label'], axis=1)
y_train = train_features_df['label']
X_test  = test_features_df.drop(['subject','label'], axis=1)
y_test  = test_features_df['label']

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("CV AUC:   ", grid.best_score_)

# ── 12. Evaluate on test set ───────────────────────────────────────────────────
best_rf = grid.best_estimator_
y_pred_proba = best_rf.predict_proba(X_test)[:,1]
y_pred       = best_rf.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test ROC AUC: {auc:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 13. Plot ROC curve ────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
