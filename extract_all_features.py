import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# — CONFIGURATION ——————————————————————————————————————————————
# CSV from earlier listing THRIVE series
SELECTED_CSV  = 'selected_series_paths.csv'
# Masks you generated via Otsu
MASK_ROOT     = 'masks_otsu'
# Output all features here
OUTPUT_CSV    = 'features_full.csv'

# PyRadiomics extractor: firstorder, GLCM, GLRLM, shape
extractor = featureextractor.RadiomicsFeatureExtractor(
    binWidth=25,
    featureClass={'firstorder': {}, 'glcm': {}, 'glrlm': {}, 'shape': {}}
)

# Frequency‐domain bands (normalized frequency ranges)
BANDS = {
    'low':  (0.0, 0.2),
    'mid':  (0.2, 0.6),
    'high': (0.6, 1.0)
}


# — HELPERS ————————————————————————————————————————————————————
def compute_frequency_features(image: sitk.Image, mask: sitk.Image):
    """Compute FFT‐based band energies, spectral entropy, mean & var amplitudes."""
    # Convert to NumPy arrays
    img_np  = sitk.GetArrayFromImage(image)   # shape (Z,Y,X)
    msk_np  = sitk.GetArrayFromImage(mask)    # same shape
    
    # Mask the image intensities
    roi     = img_np[msk_np>0].astype(np.float32)
    # Put ROI into a zero‐padded 3D array
    vol     = np.zeros_like(img_np, dtype=np.float32)
    vol[msk_np>0] = roi
    
    # 3D FFT and power spectrum
    fft     = np.fft.fftn(vol)
    ps      = np.abs(fft)**2
    # produce normalized radial frequencies grid
    shape   = vol.shape
    coords  = [np.fft.fftfreq(n) for n in shape]
    Zf, Yf, Xf = np.meshgrid(*coords, indexing='ij')
    r      = np.sqrt(Xf**2 + Yf**2 + Zf**2)  # radial freq magnitude [0,~0.866]
    r_norm = r / r.max()
    
    feats = {}
    # band energies
    for name, (lo, hi) in BANDS.items():
        mask_band = (r_norm>=lo) & (r_norm<hi)
        feats[f'freq_{name}_energy'] = ps[mask_band].sum()
    # spectral stats
    p_norm = ps / ps.sum()
    feats['freq_entropy']        = -np.sum(p_norm * np.log2(p_norm + 1e-12))
    amplitudes                   = np.abs(fft)[r_norm>0]
    feats['freq_mean_amplitude'] = amplitudes.mean()
    feats['freq_var_amplitude']  = amplitudes.var()
    return feats


# — MAIN LOOP ————————————————————————————————————————————————
df_sel = pd.read_csv(SELECTED_CSV, dtype=str)
# filter down to THRIVE SENSE
df_sel = df_sel[df_sel['SeriesName']=='THRIVE SENSE']

all_feats = []
for _, row in df_sel.iterrows():
    pid, date, series, img_dir = (
        row['NBIA_ID'], row['StudyDate'],
        row['SeriesName'], row['SeriesPath']
    )
    # load image volume
    reader = sitk.ImageSeriesReader()
    sids   = reader.GetGDCMSeriesIDs(img_dir)
    fns    = reader.GetGDCMSeriesFileNames(img_dir, sids[0])
    reader.SetFileNames(fns)
    img    = reader.Execute()
    # load mask
    mask_path = os.path.join(
        MASK_ROOT, pid, os.path.basename(img_dir),
        f"{pid}_{os.path.basename(img_dir)}_{series.replace(' ','_')}_mask.nii.gz"
    )
    if not os.path.exists(mask_path):
        print(f"[WARN] Missing mask: {mask_path}")
        continue
    msk    = sitk.ReadImage(mask_path)
    
    # a) spatial features via PyRadiomics
    rad_out = extractor.execute(img, msk)
    spatial = {k:v for k,v in rad_out.items()
               if any(k.startswith(c) for c in ('firstorder','glcm','glrlm','shape'))}
    
    # b) frequency features
    freq = compute_frequency_features(img, msk)
    
    # c) combine & annotate
    feats = {**spatial, **freq,
             'NBIA_ID': pid,
             'StudyDate': date,
             'SeriesName': series}
    all_feats.append(feats)

# — SAVE RESULTS —————————————————————————————————————————————
df_feats = pd.DataFrame(all_feats)
df_feats.to_csv(OUTPUT_CSV, index=False)
print(f"Extracted {len(df_feats)} ROIs → {OUTPUT_CSV}")
