# extract_and_train.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset
import logging
from radiomics import featureextractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib

# ── Suppress Radiomics logs ───────────────────────────────────────────────────
logging.getLogger('radiomics').setLevel(logging.ERROR)

# ── 1. Load metadata & labels ─────────────────────────────────────────────────
metadata = pd.read_csv('D:/Dataset/manifest-1542731172463/metadata.csv')
labels   = pd.read_excel('D:/Dataset/manifest-1542731172463/QIN-Breast_TreatmentResponse2014-12-16.xlsx')

# ── 2. Rename for merging ──────────────────────────────────────────────────────
metadata = metadata.rename(columns={
    'Data Description URI': 'SubjectID',
    'File Location':        'FileLocation'
})
labels = labels.rename(columns={
    'Patient ID': 'SubjectID',
    'Response':   'TreatmentResponse'
})

# ── 3. Make FileLocation absolute ──────────────────────────────────────────────
DICOM_ROOT = r'D:/Dataset/manifest-1542731172463'
metadata['FileLocation'] = (
    metadata['FileLocation']
      .str.lstrip('./\\')
      .apply(lambda rel: os.path.normpath(os.path.join(DICOM_ROOT, rel)))
)

# ── 4. Binary label ────────────────────────────────────────────────────────────
labels['label'] = labels['TreatmentResponse'].str.lower().map({'pcr':1,'non-pcr':0})

# ── 5. Merge & train/test split ───────────────────────────────────────────────
merged = metadata.merge(labels[['SubjectID','label']], on='SubjectID', how='inner')
print("Rows after merge:", len(merged))

subjects = merged['SubjectID'].unique()
train_subj, test_subj = train_test_split(
    subjects,
    test_size=0.2,
    stratify=merged.drop_duplicates('SubjectID')['label'],
    random_state=42
)

# ── 6. RadiomicsDataset ────────────────────────────────────────────────────────
class RadiomicsDataset(Dataset):
    def __init__(self, df, subj_list):
        self.df = df[df['SubjectID'].isin(subj_list)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        loc = row['FileLocation']
        # pick first .dcm in folder or the file itself
        if os.path.isdir(loc):
            files = sorted(f for f in os.listdir(loc) if f.lower().endswith('.dcm'))
            loc = os.path.join(loc, files[0])
        # read with SimpleITK
        img3 = sitk.GetArrayFromImage(sitk.ReadImage(loc))
        img  = img3[0].astype(np.float32) if img3.ndim==3 else img3.astype(np.float32)
        mask = (img > threshold_otsu(img)).astype(np.uint8)
        return {'image':img, 'mask':mask, 'label':row['label'], 'subject':row['SubjectID']}

# instantiate
train_ds = RadiomicsDataset(merged, train_subj)
test_ds  = RadiomicsDataset(merged, test_subj)
print("Train slices:", len(train_ds))
print(" Test slices:", len(test_ds))

# ── 7. Extract & save features ─────────────────────────────────────────────────
params = {'binWidth':25, 'force2D':True, 'enableCExtensions':True}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)
extractor.enableFeatureClassByName('shape2D')

def extract_df(ds, out_csv):
    recs=[]
    for i in range(len(ds)):
        s = ds[i]
        sitk_img  = sitk.GetImageFromArray(s['image'])
        sitk_mask = sitk.GetImageFromArray(s['mask'])
        feats = extractor.execute(sitk_img, sitk_mask)
        rec = {'subject':s['subject'], 'label':s['label']}
        rec.update({k:v for k,v in feats.items() if k.startswith('original_')})
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    return df

train_df = extract_df(train_ds, 'train_radiomics_features.csv')
test_df  = extract_df(test_ds,  'test_radiomics_features.csv')

# ── 8. Train Random Forest (slice-level) ───────────────────────────────────────
X_train, y_train = train_df.drop(['subject','label'], axis=1), train_df['label']
X_test,  y_test  = test_df.drop( ['subject','label'], axis=1), test_df['label']

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid = GridSearchCV(
    rf,
    {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_leaf':[1,2,4]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best RF params:", grid.best_params_)
print("Slice-level CV AUC:", grid.best_score_)

# evaluate slice-level
proba = grid.best_estimator_.predict_proba(X_test)[:,1]
print("Slice-level Test AUC:", roc_auc_score(y_test, proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, grid.best_estimator_.predict(X_test)))
print(classification_report(y_test, grid.best_estimator_.predict(X_test)))

# save model
joblib.dump(grid.best_estimator_, 'rf_slice_model.pkl')
print("Saved RF slice model to rf_slice_model.pkl")
