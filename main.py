import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pydicom                  # for DICOM I/O
import numpy as np
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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

# ── 7. Dataset class ──────────────────────────────────────────────────────────
class RadiomicsDataset(Dataset):
    def __init__(self, df, subj_list, transform=None):
        self.df = df[df['SubjectID'].isin(subj_list)].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        loc = row['FileLocation']

        # If it's a directory, pick first .dcm
        if os.path.isdir(loc):
            files = [f for f in os.listdir(loc) if f.lower().endswith('.dcm')]
            loc = os.path.join(loc, files[0])

        ds  = pydicom.dcmread(loc)
        img = ds.pixel_array.astype(np.float32)

        thresh = threshold_otsu(img)
        mask   = (img > thresh).astype(np.uint8)

        sample = {
            'image': img,
            'mask':  mask,
            'label': row['label'],
            'subject': row['SubjectID']
        }
        return self.transform(sample) if self.transform else sample

# ── 8. Instantiate & sanity-check ─────────────────────────────────────────────
train_ds = RadiomicsDataset(merged, train_subj)
test_ds  = RadiomicsDataset(merged, test_subj)

print("Train slices:", len(train_ds))
print(" Test slices:", len(test_ds))

s = train_ds[0]
print(" Keys:",  s.keys())
print(" Shape:", s['image'].shape, "Mask vals:", np.unique(s['mask']), "Label:", s['label'])

plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.imshow(s['image'], cmap='gray');    plt.title('Image');    plt.axis('off')
plt.subplot(1,2,2); plt.imshow(s['mask'],  cmap='gray');    plt.title('Otsu mask'); plt.axis('off')
plt.show()

for p in train_ds.df['FileLocation'].sample(5):
    entry = p if p.lower().endswith('.dcm') else os.path.join(p, os.listdir(p)[0])
    assert os.path.exists(entry), f"Missing file: {entry}"

# ── 9. Radiomics feature extraction ────────────────────────────────────────────
from radiomics import featureextractor
import SimpleITK as sitk
import tqdm

# Configure extractor (tweak binWidth, spacing, etc. as needed)
params = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',
    'enableCExtensions': True
}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

records = []
for i in tqdm.tqdm(range(len(train_ds)), desc="Extracting radiomics"):
    sample = train_ds[i]

    # Convert back to SimpleITK images
    sitk_img  = sitk.GetImageFromArray(sample['image'])
    sitk_mask = sitk.GetImageFromArray(sample['mask'])

    feats = extractor.execute(sitk_img, sitk_mask)

    rec = {'subject': sample['subject'], 'label': sample['label']}
    for k, v in feats.items():
        if k.startswith('original_'):
            rec[k] = v
    records.append(rec)

features_df = pd.DataFrame(records)
features_df.to_csv('train_radiomics_features.csv', index=False)
print("Saved features to train_radiomics_features.csv")

# Repeat for test set if desired…
