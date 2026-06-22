import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_afd_calculation(df, target_col, le=None, n_features=None):
    # 1. Preparation
    # Select only numeric columns first to avoid string columns turning into NaN columns
    X_df = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    
    # Ensure they are numeric (force conversion again just in case)
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    
    X_df = X_df.dropna()
    
    if X_df.shape[1] == 0:
        raise ValueError("Aucune variable numérique valide trouvée. Vérifiez que vous avez sélectionné des colonnes contenant des nombres.")
        
    # Sync with target
    y = df.loc[X_df.index, target_col].values
    X = X_df.values
    
    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)
        
    classes = le.classes_
    n_classes = len(classes)
    
    if n_classes < 2:
        raise ValueError("L'analyse factorielle discriminante nécessite au moins 2 classes distinctes.")
    
    if n_features is None:
        n_features = X.shape[1]
        
    # 2. Scatter Matrices
    global_mean = np.mean(X, axis=0)
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    
    for i in range(n_classes):
        X_k = X[y_encoded == i]
        if len(X_k) == 0: continue
        
        mean_k = np.mean(X_k, axis=0)
        
        # Within-class
        diff_w = X_k - mean_k
        S_W += diff_w.T @ diff_w
        
        # Between-class
        n_k = X_k.shape[0]
        mean_diff = (mean_k - global_mean).reshape(-1, 1)
        S_B += n_k * (mean_diff @ mean_diff.T)
    
    # 3. Solve Eigen Problem (S_W^-1 * S_B)
    # Adding a small epsilon for stability
    epsilon = 1e-6
    S_W_reg = S_W + np.eye(n_features) * epsilon
    
    try:
        inv_S_W = np.linalg.inv(S_W_reg)
    except np.linalg.LinAlgError:
        inv_S_W = np.linalg.pinv(S_W_reg)
        
    eig_vals, eig_vecs = np.linalg.eig(inv_S_W @ S_B)
    
    # 4. Sort and select axes
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real
    
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    # Select first q axes (q <= min(K-1, p))
    q = min(n_classes - 1, n_features)
    # We need at least 1 axis to plot
    if q < 1: q = 1
    
    W_axes = eig_vecs[:, :q]
    
    # 5. Projection
    X_afd = X @ W_axes
    
    # 6. Formatting results
    df_afd = pd.DataFrame(X_afd, columns=[f"LD{i+1}" for i in range(q)], index=X_df.index)
    df_afd['Target'] = y
    
    return {
        "df_afd": df_afd,
        "eig_vals": eig_vals[:q].tolist(),
        "eig_vecs": eig_vecs[:, :q].tolist(), # All eigenvectors for the selected axes
        "W_axes": W_axes,
        "S_W": S_W.tolist(),
        "S_B": S_B.tolist(),
        "classes": classes.tolist(),
        "feature_names": X_df.columns.tolist(),
        "y_encoded": y_encoded,
        "le": le
    }

def simulate_app_logic(filepath, target_col, feature_cols, test_size=0.2):
    print("--- Simulating Original app.py Logic ---")
    df_raw = pd.read_csv(filepath)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Case-insensitive target lookup
    if target_col not in df_raw.columns:
        col_lower_map = {c.lower(): c for c in df_raw.columns}
        if target_col.lower() in col_lower_map:
            target_col = col_lower_map[target_col.lower()]
        else:
            print(f"Error: target {target_col} not found")
            return
            
    col_map = {c.lower(): c for c in df_raw.columns}
    player_col = col_map.get('player')
    target_col_actual = col_map.get(target_col.lower()) or target_col
    
    extra_cols = []
    if player_col: extra_cols.append(player_col)
    if 'image_url' in col_map: extra_cols.append(col_map['image_url'])
        
    cols_to_use = [target_col_actual] + feature_cols + extra_cols
    cols_to_use = [c for c in cols_to_use if c in df_raw.columns]
    
    # FORCED NUMERIC CONVERSION for features (original logic)
    df_raw_original = df_raw.copy()
    for col in feature_cols:
        if col in df_raw_original.columns:
            df_raw_original[col] = pd.to_numeric(df_raw_original[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    essential_cols = [target_col_actual] + feature_cols
    df_original_processed = df_raw_original[cols_to_use].dropna(subset=essential_cols).copy()
    print(f"Shape of df after original dropna: {df_original_processed.shape}")
    
    # Simulating fixed logic
    print("\n--- Simulating Fixed app.py Logic ---")
    df_raw_fixed = df_raw.copy()
    
    # Clean feature_cols: exclude target, player, and image_url
    exclude_cols = {target_col_actual.lower()}
    if player_col:
        exclude_cols.add(player_col.lower())
    if 'image_url' in col_map:
        exclude_cols.add(col_map['image_url'].lower())
        
    cleaned_feature_cols = [c for c in feature_cols if c.lower() not in exclude_cols]
    
    # Select only numeric feature columns
    valid_feature_cols = []
    for col in cleaned_feature_cols:
        if col in df_raw_fixed.columns:
            converted = pd.to_numeric(df_raw_fixed[col].astype(str).str.replace(',', '.'), errors='coerce')
            if converted.notna().sum() > 0.5 * len(converted):
                df_raw_fixed[col] = converted
                valid_feature_cols.append(col)
                
    print(f"Original features count: {len(feature_cols)}")
    print(f"Cleaned valid features count: {len(valid_feature_cols)}")
    print(f"Excluded features: {[c for c in feature_cols if c not in valid_feature_cols]}")
    
    cols_to_use_fixed = [target_col_actual] + valid_feature_cols + extra_cols
    cols_to_use_fixed = [c for c in cols_to_use_fixed if c in df_raw_fixed.columns]
    
    essential_cols_fixed = [target_col_actual] + valid_feature_cols
    df_fixed_processed = df_raw_fixed[cols_to_use_fixed].dropna(subset=essential_cols_fixed).copy()
    print(f"Shape of df after fixed dropna: {df_fixed_processed.shape}")
    
    if df_fixed_processed.empty:
        print("Fixed df is empty!")
        return
        
    # Position Mapping
    if target_col_actual.lower() == 'pos':
        def map_position(pos):
            pos = str(pos).upper()
            if 'GK' in pos or 'DF' in pos:
                return 'Defender'
            elif 'MF' in pos:
                return 'Milieu'
            elif 'FW' in pos:
                return 'Attaque'
            return 'Other'
        df_fixed_processed[target_col_actual] = df_fixed_processed[target_col_actual].apply(map_position)
        df_fixed_processed = df_fixed_processed[df_fixed_processed[target_col_actual] != 'Other']
        print(f"Shape after position mapping: {df_fixed_processed.shape}")
        
    # Filter out classes with only 1 member
    class_counts = df_fixed_processed[target_col_actual].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    if len(valid_classes) < len(class_counts):
        df_fixed_processed = df_fixed_processed[df_fixed_processed[target_col_actual].isin(valid_classes)].copy()
        class_counts = df_fixed_processed[target_col_actual].value_counts()
        print(f"Filtered to classes with >= 2 members. New shape: {df_fixed_processed.shape}")

    print("Class counts:\n", class_counts)
    can_stratify = class_counts.min() >= 2
    
    train_df, test_df = train_test_split(
        df_fixed_processed, 
        test_size=test_size, 
        stratify=df_fixed_processed[target_col_actual] if can_stratify else None, 
        random_state=42
    )
    
    le = LabelEncoder()
    le.fit(df_fixed_processed[target_col_actual])
    
    # Run calculation
    afd_train = run_afd_calculation(train_df, target_col_actual, le=le, n_features=len(valid_feature_cols))
    print("AFD run successfully on train_df!")
    print(f"Eigenvalues: {afd_train['eig_vals']}")

if __name__ == '__main__':
    csv_file = r"c:\Users\DZ Laptops\Desktop\ACP\premier_league_players_cleaned.csv"
    # Read headers
    df = pd.read_csv(csv_file)
    headers = list(df.columns)
    simulate_app_logic(csv_file, "Pos", headers)
