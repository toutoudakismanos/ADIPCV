import os
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset
import logging

# ➀ Suppress Radiomics logs
logging.getLogger('radiomics').setLevel(logging.ERROR)

# ➁ Use a non-GUI backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from radiomics import featureextractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# ── 1. Load metadata and labels ───────────────────────────────────────────────
metadata = pd.read_csv(
    'D:/Dataset/manifest-1542731172463/metadata.csv'
)
labels_df = pd.read_excel(
    'D:/Dataset/manifest-1542731172463/QIN-Breast_TreatmentResponse2014-12-16.xlsx'
)

# ── 2. Rename columns for merging ──────────────────────────────────────────────
metadata = metadata.rename(columns={
    'Data Description URI': 'SubjectID',
    'File Location':        'FileLocation'
})
labels_df = labels_df.rename(columns={
    'Patient ID': 'SubjectID',
    'Response':   'TreatmentResponse'
})

# ── 3. Fix up FileLocation to be absolute paths ────────────────────────────────
DICOM_ROOT = r'D:/Dataset/manifest-1542731172463'
metadata['FileLocation'] = (
    metadata['FileLocation']
      .str.lstrip('./\\')
      .apply(lambda rel: os.path.normpath(os.path.join(DICOM_ROOT, rel)))
)

# ── 4. Create binary label ─────────────────────────────────────────────────────
labels_df['label'] = (
    labels_df['TreatmentResponse']
      .str.lower()
      .map({'pcr': 1, 'non-pcr': 0})
)

# ── 5. Merge and sanity-check ─────────────────────────────────────────────────
merged = metadata.merge(
    labels_df[['SubjectID','label']],
    on='SubjectID', how='inner'
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

# ── 7. Dataset class reading only one DICOM per folder ────────────────────────
class RadiomicsDataset(Dataset):
    def __init__(self, df, subj_list, transform=None):
        self.df = df[df['SubjectID'].isin(subj_list)].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        loc = row['FileLocation']

        # If it's a directory, pick the first .dcm inside
        if os.path.isdir(loc):
            files = sorted(f for f in os.listdir(loc)
                           if f.lower().endswith('.dcm'))
            if not files:
                raise FileNotFoundError(f"No DICOMs in {loc}")
            input_path = os.path.join(loc, files[0])
        else:
            input_path = loc

        # Read exactly one slice with SimpleITK
        sitk_img = sitk.ReadImage(input_path)
        arr = sitk.GetArrayFromImage(sitk_img)
        img = arr[0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)

        # Otsu-based mask
        thresh = threshold_otsu(img)
        mask   = (img > thresh).astype(np.uint8)

        sample = {
            'image':   img,
            'mask':    mask,
            'label':   row['label'],
            'subject': row['SubjectID']
        }
        return self.transform(sample) if self.transform else sample

# ── 8. Instantiate & quick sanity-check ───────────────────────────────────────
train_ds = RadiomicsDataset(merged, train_subj)
test_ds  = RadiomicsDataset(merged, test_subj)

print("Train slices:", len(train_ds))
print(" Test slices:", len(test_ds))

s = train_ds[0]
print("Keys: ",  s.keys())
print("Shape:", s['image'].shape,
      "Mask vals:", np.unique(s['mask']),
      "Label:", s['label'])

# Save a sanity-check figure
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(s['image'], cmap='gray')
plt.title('Image'); plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(s['mask'],  cmap='gray')
plt.title('Otsu mask'); plt.axis('off')
plt.savefig('sanity_check.png', bbox_inches='tight')
plt.close()
print("Saved sanity-check image to sanity_check.png")

# ── 9. Radiomics extraction (train & test) ───────────────────────────────────
# Configure for 2D + shape2D
params = {
    'binWidth': 25,
    'force2D': True,
    'enableCExtensions': True
}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)
extractor.enableFeatureClassByName('shape2D')

def extract_df(dataset, out_csv):
    records = []
    for i in range(len(dataset)):
        sample = dataset[i]
        sitk_img  = sitk.GetImageFromArray(sample['image'])
        sitk_mask = sitk.GetImageFromArray(sample['mask'])
        feats = extractor.execute(sitk_img, sitk_mask)
        rec = {'subject': sample['subject'], 'label': sample['label']}
        # grab only the computed feature entries
        rec.update({k: v for k, v in feats.items() if k.startswith('original_')})
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv}")
    return df

train_df = extract_df(train_ds, 'train_radiomics_features.csv')
test_df  = extract_df(test_ds,  'test_radiomics_features.csv')

# ── 10. Train & evaluate model ────────────────────────────────────────────────
X_train = train_df.drop(['subject','label'], axis=1)
y_train = train_df['label']

X_test  = test_df.drop(['subject','label'], axis=1)
y_test  = test_df['label']

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid = GridSearchCV(
    rf,
    {'n_estimators':[100,200], 'max_depth':[None,10,20],
     'min_samples_leaf':[1,2,4]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("CV AUC:   ", grid.best_score_)

best_rf = grid.best_estimator_
y_proba = best_rf.predict_proba(X_test)[:,1]
y_pred  = best_rf.predict(X_test)

auc = roc_auc_score(y_test, y_proba)
print(f"Test ROC AUC: {auc:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ── 11. Plot & save ROC ──────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png', bbox_inches='tight')
plt.close()
print("Saved ROC curve to roc_curve.png")
