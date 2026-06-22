import pandas as pd
import numpy as np
import os
import re
import sys

def verify_and_clean_csv(input_path, output_path):
    print(f"--- Advanced Cleaning & Type Verification for {input_path} ---")
    
    df = pd.read_csv(input_path)
    
    # 1. Clean Column Names
    new_columns = []
    for col in df.columns:
        clean_name = re.sub(r'^Unnamed: \d+_level_0_', '', col)
        new_columns.append(clean_name)
    df.columns = new_columns
    
    # Standardize column mapping to handle both Upper and lower cases based on what is in FBref
    col_map = {c.lower(): c for c in df.columns}
    
    # 2. Remove intermediate header rows
    player_col = col_map.get('player')
    rk_col = col_map.get('rk') or col_map.get('ranker')
    if player_col:
        df = df[df[player_col] != 'player']
        df = df[df[player_col] != 'Player']
    if rk_col:
        df = df[df[rk_col] != 'Rk']
        df = df[df[rk_col] != 'ranker']
        
    # 3. Handle 'Age' column
    age_col = col_map.get('age')
    if age_col:
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
        df[age_col] = df[age_col].apply(convert_age)

    # 4. Clean 'Nation'
    nation_col = col_map.get('nation') or col_map.get('nationality')
    if nation_col:
        df[nation_col] = df[nation_col].astype(str).str.split(' ').str[-1]

    # 5. Drop irrelevant columns
    cols_to_drop = [
        'Matches', 'Player_URL', 'Matches_URL', 'Rk', 'ranker', 'matches',
        'Unnamed: 24_level_0_Matches'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 6. Type Verification and Enforcement
    categorical_cols = ['player', 'nation', 'nationality', 'pos', 'position', 'squad', 'team', 'comp', 'image_url', 'born', 'birth_year', 'player_url']
    int_cols = ['starts', 'games_starts', 'min', 'minutes', 'gls', 'goals', 'ast', 'assists', 'g+a', 'goals_assists', 'g-pk', 'goals_pens', 'pk', 'pens_made', 'pkatt', 'pens_att', 'crdy', 'cards_yellow', 'crdr', 'cards_red', '90s', 'games']
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in categorical_cols:
            df[col] = df[col].astype(str).replace(['nan', 'NaN'], np.nan)
        elif col_lower in int_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
    # 7. Drop rows where 'Player' is NaN
    if player_col and player_col in df.columns:
        df = df.dropna(subset=[player_col])

    # 8. Reset Index
    df = df.reset_index(drop=True)
    
    # Print statistics on types
    print("\n[Types Verified]")
    print(df.dtypes)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Nettoyé avec vérification stricte des types sauvegardé dans: {output_path}")
    print(f"📊 Dimensions finales: {df.shape}")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Default fallback
        input_file = "l_photos.csv"
        output_file = "l2_cleaned.csv"
    
    if os.path.exists(input_file):
        verify_and_clean_csv(input_file, output_file)
    else:
        print(f"❌ Erreur: {input_file} introuvable.")
