import pandas as pd
import json
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simulate_frontend_parse(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simulate JS split logic
    lines = text.split('\n')
    sep = ';' if ';' in lines[0] else ','
    originalRows = []
    for line in lines:
        line_trimmed = line.strip()
        if not line_trimmed: continue
        row = [cell.strip() for cell in line_trimmed.split(sep)]
        if any(c != "" for c in row):
            originalRows.append(row)
            
    return originalRows

def simulate_backend(originalRows):
    # Simulate JS submit
    allHeaders = originalRows[0]
    dataRows = originalRows[1:]
    
    # Frontend active chips (suppose all are active)
    activeColIndices = list(range(len(allHeaders)))
    
    finalData = [allHeaders]
    for row in dataRows:
        finalData.append([row[i] if i < len(row) else "" for i in activeColIndices])
        
    # Backend load
    headers = finalData[0]
    n_cols = len(headers)
    rows = [row[:n_cols] for row in finalData[1:]]
    df_raw = pd.DataFrame(rows, columns=headers)
    
    # backend cleaning
    df_raw.columns = df_raw.columns.str.strip()
    target_col = 'position'
    
    # case-insensitive
    col_lower_map = {c.lower(): c for c in df_raw.columns}
    target_col_actual = col_lower_map.get(target_col.lower()) or target_col
    player_col = col_lower_map.get('player')
    
    exclude_cols = {target_col_actual.lower()}
    if player_col:
        exclude_cols.add(player_col.lower())
    if 'image_url' in col_map := {c.lower(): c for c in df_raw.columns}:
        exclude_cols.add(col_map['image_url'].lower())
        
    feature_cols = [c for c in headers if c.lower() not in exclude_cols]
    
    cleaned_feature_cols = [c for c in feature_cols if c.lower() not in exclude_cols]
    
    # Filter only numeric feature columns
    valid_feature_cols = []
    for col in cleaned_feature_cols:
        if col in df_raw.columns:
            converted = pd.to_numeric(df_raw[col].astype(str).str.replace(',', '.'), errors='coerce')
            print(f"Col {col}: non-NaNs = {converted.notna().sum()} / {len(converted)}")
            if converted.notna().sum() > 0.5 * len(converted):
                df_raw[col] = converted
                valid_feature_cols.append(col)
                
    print("Valid feature cols:", valid_feature_cols)
    
    extra_cols = []
    if player_col: extra_cols.append(player_col)
    if 'player_url' in col_map: extra_cols.append(col_map['player_url']) # extra URLs
    
    cols_to_use = [target_col_actual] + valid_feature_cols + extra_cols
    cols_to_use = [c for c in cols_to_use if c in df_raw.columns]
    
    essential_cols = [target_col_actual] + valid_feature_cols
    df = df_raw[cols_to_use].dropna(subset=essential_cols).copy()
    print("Df shape after dropna:", df.shape)
    
if __name__ == '__main__':
    csv_file = r"c:\Users\DZ Laptops\Desktop\ACP\t_raw.csv"
    originalRows = simulate_frontend_parse(csv_file)
    simulate_backend(originalRows)
