import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

def run_afd_calculation(df, target_col, le=None):
    # 1. Preparation
    # Select only numeric columns first to avoid string columns turning into NaN columns after numeric conversion
    X_df = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    
    # Ensure they are numeric (force conversion again just in case)
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    
    # Drop any rows with NaNs
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

def calculate_afd_metrics(y_true, y_pred, y_prob=None, classes=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC Curve for each class (One-vs-Rest)
    # This requires probabilities or scores, which we don't have directly from LDA projection 
    # unless we use a classifier (like Centroid-based or K-NN on projected space).
    # For now, let's just return the basics.
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
