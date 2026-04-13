"""Quick sanity check for both EuroSAT datasets."""
import os, pandas as pd
from PIL import Image
import rasterio

ROOT = r'd:\EuroSat Classification Final Project\datasets'

# --- RGB ---
df = pd.read_csv(os.path.join(ROOT, 'EuroSAT', 'train.csv'))
if df.columns[0] not in ('Filename', 'filename'):
    df = df.iloc[:, 1:]
row = df.iloc[0]
img = Image.open(os.path.join(ROOT, 'EuroSAT', row['Filename'])).convert('RGB')
print(f"RGB : {row['Filename']}  size={img.size}  label={row['Label']}  OK")

# --- Multispectral ---
df2 = pd.read_csv(os.path.join(ROOT, 'EuroSATallBands', 'train.csv'))
row2 = df2.iloc[0]
with rasterio.open(os.path.join(ROOT, 'EuroSATallBands', row2['Filename'])) as src:
    arr = src.read()
print(f"TIF : {row2['Filename']}  shape={arr.shape}  label={row2['Label']}  OK")

# --- Split sizes ---
print()
for ds in ['EuroSAT', 'EuroSATallBands']:
    for split in ['train', 'validation', 'test']:
        path = os.path.join(ROOT, ds, f'{split}.csv')
        n = len(pd.read_csv(path))
        print(f"  {ds:<22} {split:<12} {n:>6} rows")

print("\nAll checks passed.")
