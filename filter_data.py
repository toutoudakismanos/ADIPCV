import os
import pandas as pd

# 1. Configuration
DICOM_ROOT = r'D:/Dataset/manifest-1554746803134/QIN-BREAST-02'
TARGET_SERIES = {'DWIEPIb0200800', 'multi', 'THRIVE SENSE'}

# 2. Traverse and collect
records = []
for patient_id in os.listdir(DICOM_ROOT):
    patient_path = os.path.join(DICOM_ROOT, patient_id)
    if not os.path.isdir(patient_path):
        continue

    for study_folder in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_folder)
        if not os.path.isdir(study_path):
            continue

        for series_folder in os.listdir(study_path):
            series_path = os.path.join(study_path, series_folder)
            if not os.path.isdir(series_path):
                continue

            # Strip any leading/trailing dashes, split on '-' up to 2 splits:
            #   e.g. "7501.000000-DWIEPIb0200800-76277" → ["7501.000000", "DWIEPIb0200800", "76277"]
            name_core = series_folder.strip('-')
            parts = name_core.split('-', 2)
            if len(parts) >= 2:
                desc = parts[1]
            else:
                desc = name_core  # fallback

            if desc in TARGET_SERIES:
                records.append({
                    'NBIA_ID':    patient_id,
                    'StudyDate':  study_folder,
                    'SeriesName': desc,
                    'SeriesPath': series_path
                })

# 3. Build DataFrame
df_selected = pd.DataFrame(records,
    columns=['NBIA_ID', 'StudyDate', 'SeriesName', 'SeriesPath'])

# 4. Check for any patients missing one or more series
#    Pivot so each series is a column, then look for NaNs
pivot = df_selected.pivot_table(
    index='NBIA_ID', columns='SeriesName', 
    values='SeriesPath', aggfunc='first'
)
missing_counts = pivot.isna().sum(axis=1)
if missing_counts.any():
    print("Patients missing at least one of the target series:")
    print(missing_counts[missing_counts > 0])

# 5. Save results
df_selected.to_csv('selected_series_paths.csv', index=False)
print(f"Wrote selected_series_paths.csv with {len(df_selected)} rows.")
