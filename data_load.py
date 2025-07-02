import os
import pandas as pd

# 1. Paths — adjust if needed
EXCEL_PATH = 'QIN-BREAST-02_clinicalData-Transformed-20191022-Revised20200210.xlsx'  
DICOM_ROOT = r'D:/Dataset/manifest-1554746803134/QIN-BREAST-02'

# 2. Read clinical sheet
df = pd.read_excel(EXCEL_PATH, sheet_name=0, dtype=str)

# 3. Drop NOMATCH
df = df[df['Response'] != 'NOMATCH'].copy()

# 4. Binarize PCR→1, others→0
df['label'] = (df['Response'] == 'PCR').astype(int)

# 5. Build folder paths & warn if missing
folder_paths = []
missing = []
for pid in df['NBIA ID']:
    pth = os.path.join(DICOM_ROOT, pid)
    if not os.path.isdir(pth):
        missing.append(pid)
        folder_paths.append(None)
    else:
        folder_paths.append(pth)
df['folder_path'] = folder_paths

if missing:
    print(f"Warning: no DICOM folder found for {len(missing)} IDs:\n", missing)

# 6. Select additional covariates if present
covs = ['Size (cm)', 'Grade', 'ER Status', 'PR Status', 'HER2 Status', 'Clinical Stage']
present_covs = [c for c in covs if c in df.columns]
output_cols = ['NBIA ID', 'folder_path', 'label'] + present_covs

# 7. Write out CSV
df[output_cols].to_csv('labels.csv', index=False)
print("Wrote labels.csv with columns:", output_cols)
