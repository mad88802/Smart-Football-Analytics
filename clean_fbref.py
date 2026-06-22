import pandas as pd
import numpy as np
import os
import re

def clean_fbref_csv(input_path, output_path):
    print(f"--- Cleaning {input_path} ---")
    
    # Load the CSV
    # FBref CSVs often have multiple header rows or flattened multi-index headers
    df = pd.read_csv(input_path)
    
    # 1. Clean Column Names
    # Remove "Unnamed: X_level_0_" prefixes
    new_columns = []
    for col in df.columns:
        # Match pattern "Unnamed: ... _level_0_" followed by the actual name
        clean_name = re.sub(r'^Unnamed: \d+_level_0_', '', col)
        new_columns.append(clean_name)
    df.columns = new_columns
    
    # 2. Remove intermediate header rows
    # These rows repeat the header names (e.g., 'Player' or 'Rk')
    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player']
    if 'Rk' in df.columns:
        df = df[df['Rk'] != 'Rk']
    
    # 3. Handle 'Age' column (Format: '25-132' -> years as float)
    # 132 days / 365.25 days/year ≈ 0.36 years
    if 'Age' in df.columns:
        def convert_age(age_str):
            if pd.isna(age_str) or not isinstance(age_str, str):
                return age_str
            parts = age_str.split('-')
            if len(parts) == 2:
                try:
                    years = float(parts[0])
                    days = float(parts[1])
                    return round(years + (days / 365.25), 2)
                except ValueError:
                    return age_str
            return age_str
        
        df['Age'] = df['Age'].apply(convert_age)

    # 4. Clean 'Nation' (Format: 'us USA' -> 'USA')
    if 'Nation' in df.columns:
        df['Nation'] = df['Nation'].str.split(' ').str[-1]

    # 5. Drop irrelevant columns
    cols_to_drop = [
        'Matches', 'Player_URL', 'Matches_URL', 'Rk', 
        'Unnamed: 24_level_0_Matches'
    ]
    # Also drop any columns that are entirely NaN
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 6. Convert numeric columns to float/int
    # Skip non-numeric columns like Player, Nation, Pos, Squad
    categorical_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Born', 'Image_URL']
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 7. Drop rows where 'Player' is NaN (summary rows often have NaN in Player)
    if 'Player' in df.columns:
        df = df.dropna(subset=['Player'])

    # 8. Reset Index
    df = df.reset_index(drop=True)
    
    # Save Cleaned CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Cleaned data saved to: {output_path}")
    print(f"📊 Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    input_file = "l_photos.csv"
    output_file = "l2_cleaned.csv"
    
    if os.path.exists(input_file):
        clean_fbref_csv(input_file, output_file)
    else:
        print(f"❌ Error: {input_file} not found.")
