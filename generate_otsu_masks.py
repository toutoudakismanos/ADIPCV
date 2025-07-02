import os
import pandas as pd
import SimpleITK as sitk

# Configuration
SELECTED_CSV = 'selected_series_paths.csv'  # CSV from previous step
OUTPUT_MASK_ROOT = 'masks_otsu'            # Directory to save generated masks

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_MASK_ROOT, exist_ok=True)

# Load the selected series list
df = pd.read_csv(SELECTED_CSV, dtype=str)
# Filter for THRIVE SENSE series only
thrive_df = df[df['SeriesName'] == 'THRIVE SENSE']

# Process each THRIVE SENSE series
for idx, row in thrive_df.iterrows():
    pid = row['NBIA_ID']
    study = row['StudyDate']
    series_name = row['SeriesName']
    series_path = row['SeriesPath']
    
    # Read the DICOM series into a volume
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_path)
    if not series_ids:
        print(f"[ERROR] No DICOM series found in {series_path}")
        continue
    file_names = reader.GetGDCMSeriesFileNames(series_path, series_ids[0])
    reader.SetFileNames(file_names)
    image = reader.Execute()
    
    # Step 1: Otsu thresholding to create initial mask
    mask_otsu = sitk.OtsuThreshold(image, 0, 1)
    
    # Step 2: Keep only the largest connected component
    cc = sitk.ConnectedComponent(mask_otsu)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest_cc = cc == 1
    
    # Step 3: Morphological closing to fill holes (radius=2 voxels)
    closed_mask = sitk.BinaryMorphologicalClosing(largest_cc, [2] * image.GetDimension())
    
    # Prepare output path
    study_folder_name = os.path.basename(series_path)
    mask_dir = os.path.join(OUTPUT_MASK_ROOT, pid, study_folder_name)
    os.makedirs(mask_dir, exist_ok=True)
    mask_filename = f"{pid}_{study_folder_name}_{series_name.replace(' ', '_')}_mask.nii.gz"
    mask_path = os.path.join(mask_dir, mask_filename)
    
    # Save the mask as NIfTI
    sitk.WriteImage(closed_mask, mask_path)
    print(f"Generated mask for {pid} ({series_name}) at:\n  {mask_path}")


