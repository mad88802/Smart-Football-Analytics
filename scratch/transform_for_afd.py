import pandas as pd
import numpy as np
import re

# Load the original CSV
df = pd.read_csv(r"c:\Users\DZ Laptops\Desktop\ACP\premier_league_players_cleaned.csv")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample Pos values:", df['Pos'].unique()[:15])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Drop columns that are NOT useful for AFD
# ─────────────────────────────────────────────────────────────────────────────
# Drop: Player (name), Nation, Squad (categorical text), Born (redundant with Age),
#        Image_URL (irrelevant)
df_afd = df.drop(columns=['Player', 'Nation', 'Squad', 'Born', 'Image_URL'], errors='ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Clean the 'Pos' column → AFD target class (simplified to 3 classes)
# ─────────────────────────────────────────────────────────────────────────────
def simplify_position(pos):
    """Map position strings to 3 clean classes for AFD."""
    if pd.isna(pos):
        return None
    pos = str(pos).strip()
    # Take first position if multiple (e.g. "MF,FW" → "MF")
    first = pos.split(',')[0].strip()
    mapping = {
        'GK': 'Gardien',
        'DF': 'Defenseur',
        'MF': 'Milieu',
        'FW': 'Attaquant',
    }
    return mapping.get(first, None)

df_afd['Position'] = df_afd['Pos'].apply(simplify_position)

# Drop rows where position could not be mapped
before = len(df_afd)
df_afd = df_afd.dropna(subset=['Position'])
print(f"\nDropped {before - len(df_afd)} rows with unmappable positions")

# Drop the original 'Pos' column (we keep 'Position' as the class label)
df_afd = df_afd.drop(columns=['Pos'])

# ─────────────────────────────────────────────────────────────────────────────
# 3. Ensure all other columns are numeric
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = [c for c in df_afd.columns if c != 'Position']
for col in numeric_cols:
    df_afd[col] = pd.to_numeric(df_afd[col], errors='coerce')

# Drop rows with any NaN in numeric columns
before = len(df_afd)
df_afd = df_afd.dropna(subset=numeric_cols)
print(f"Dropped {before - len(df_afd)} rows with NaN numeric values")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Class distribution check
# ─────────────────────────────────────────────────────────────────────────────
print("\nClass distribution:")
print(df_afd['Position'].value_counts())
print(f"\nFinal shape: {df_afd.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Move 'Position' to the LAST column (AFD expects target at end)
# ─────────────────────────────────────────────────────────────────────────────
cols = [c for c in df_afd.columns if c != 'Position'] + ['Position']
df_afd = df_afd[cols]

print("\nFinal columns:", df_afd.columns.tolist())
print("\nFirst 3 rows:")
print(df_afd.head(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
output_path = r"c:\Users\DZ Laptops\Desktop\ACP\premier_league_afd_ready.csv"
df_afd.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")
