import os
import itertools
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from groq import Groq
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import json
import math
import zipfile
import shutil
from tableauhyperapi import (HyperProcess, Connection, TableDefinition, 
                             SqlType, Telemetry, Inserter, CreateMode, TableName)

load_dotenv()

# Fix Windows charmap issue and Colorama crashes
import builtins
def safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except OSError:
        pass # Ignore colorama Errno 22 crashes

print = safe_print

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

IMG_DIR = "static"
os.makedirs(IMG_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/acp")
def acp_home():
    return render_template("upload.html")


@app.route("/afd")
def afd_home():
    return render_template("upload_afd.html")



@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get('csvfile')
    edited_data = request.form.get('edited_data')
    acp_type = request.form.get('acp_type')
    data_type = request.form.get('data_type')
    criterion = request.form.get('criterion')
    api_key = request.form.get('api_key') or os.getenv('GROQ_API_KEY')
    
    df = None
    
    try:
        # Prioritize edited/filtered data from the frontend
        if edited_data:
            try:
                data = json.loads(edited_data)
                if data and len(data) > 0:
                    df = pd.DataFrame(data[1:], columns=data[0])
            except Exception as e:
                print(f"DEBUG: Error parsing edited_data: {e}")

        # Fallback to uploaded file if no edited_data or parsing failed
        if df is None and uploaded_file and uploaded_file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filepath)
            df = pd.read_csv(filepath)
            if df.shape[1] < 2:
                try:
                    df_sep = pd.read_csv(filepath, sep=';')
                    if df_sep.shape[1] > df.shape[1]: df = df_sep
                except: pass
        
        if df is None:
            return redirect(url_for('index'))
            
        # Prepare potential identifier columns (names, etc.)
        identifier_df = df.select_dtypes(exclude=[np.number])
        
        # Numeric conversion logic
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(r'(\d),(\d{3})(?!\d)', r'\1\2', regex=True)
                    cleaned = cleaned.str.replace(',', '.')
                    converted = pd.to_numeric(cleaned, errors='coerce')
                    # If conversion is mostly successful, it's likely a numeric column stored as text
                    if converted.isna().sum() < (len(df) * 0.5): 
                        df[col] = converted
                except: pass
        
        # After conversion, re-examine potential identifiers
        identifier_df = df.select_dtypes(exclude=[np.number])
        
        # Copy original for reference
        df_original = df.copy()
        
        # Keeping only numeric for PCA
        df = df.select_dtypes(include=[np.number])
        if df.empty: return "Erreur: Aucune colonne numérique trouvée."
        
        df = df.astype(float)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
        if df.isnull().values.any():
            df = df.fillna(df.mean())
            df = df.dropna(axis=1, how='all')
            # If we drop rows, we must drop them from identifier_df too
            initial_indices = df.index
            df = df.dropna(axis=0, how='any')
            if not identifier_df.empty:
                identifier_df = identifier_df.loc[df.index]
            
        if df.empty: return "Erreur: Données vides après nettoyage."
            
    except Exception as e: return f"Erreur de lecture/traitement des données: {e}"
    
    # Selection of the best identifier column
    if not identifier_df.empty:
        # Prioritize columns that look like names (contain 'player', 'name', 'nom')
        name_cols = [c for c in identifier_df.columns if any(x in c.lower() for x in ['player', 'name', 'nom', 'individu'])]
        if name_cols:
            id_col = name_cols[0]
        else:
            id_col = identifier_df.columns[0]
            
        # Ensure index is 1D even if duplicate columns exist
        idx_data = identifier_df[id_col]
        if hasattr(idx_data, 'ndim') and idx_data.ndim > 1:
            idx_data = idx_data.iloc[:, 0]
            
        df.index = idx_data
        df.index.name = id_col
    else:
        df.index = [f"Ind{i+1}" for i in range(len(df))]
        df.index.name = "Individus"
    
    try:
        # Handle Image_URL carefully in case of duplicates
        img_urls = None
        if 'Image_URL' in identifier_df.columns:
            img_data = identifier_df['Image_URL']
            if hasattr(img_data, 'ndim') and img_data.ndim > 1:
                img_data = img_data.iloc[:, 0]
            img_urls = img_data
        results = run_pca(df, acp_type, data_type, criterion, img_urls, api_key)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("CRITICAL ACP ERROR:", tb)
        return f"Erreur calcul ACP: {e}<br><pre>{tb}</pre>"
    
    return render_template("results.html", 
                           df=df.to_html(classes='table', border=0),
                           acp_type=acp_type, data_type=data_type, criterion=criterion, **results)

@app.route("/analyze_afd", methods=["POST"])
def analyze_afd():
    target_col = (request.form.get('target_col') or '').strip()
    feature_cols = [c.strip() for c in request.form.getlist('feature_cols') if c and c.strip()]
    try:
        test_size = float(request.form.get('test_size', 0.2))
    except (TypeError, ValueError):
        test_size = 0.2
    test_size = min(max(test_size, 0.1), 0.5)
    api_key = request.form.get('api_key') or os.getenv('GROQ_API_KEY')

    def normalize_key(value):
        return str(value).strip().lower()

    def resolve_column(name, columns):
        col_map = {normalize_key(c): c for c in columns}
        resolved = col_map.get(normalize_key(name))
        if resolved:
            return resolved
        pos_aliases = {'pos': 'position', 'position': 'pos'}
        return col_map.get(pos_aliases.get(normalize_key(name), ''))

    def coerce_numeric_series(series):
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.str.replace('\u00a0', '', regex=False).str.replace(' ', '', regex=False)
        cleaned = cleaned.str.replace(r'(\d),(\d{3})(?!\d)', r'\1\2', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    
    edited_data_json = request.form.get('edited_data')
    if edited_data_json:
        try:
            data = json.loads(edited_data_json)
            headers = data[0]
            n_cols = len(headers)
            # Truncate rows to match header count (trailing commas create extra empty columns)
            rows = [(row + [''] * n_cols)[:n_cols] for row in data[1:]]
            df_raw = pd.DataFrame(rows, columns=headers)
        except Exception as e:
            return f"Erreur lors de la lecture des données éditées: {e}"
    else:
        uploaded_file = request.files.get('csvfile')
        if not uploaded_file or uploaded_file.filename == '':
            return redirect(url_for('afd_home'))
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filepath)
        df_raw = pd.read_csv(filepath)
        if df_raw.shape[1] < 2:
            try:
                df_sep = pd.read_csv(filepath, sep=';')
                if df_sep.shape[1] > df_raw.shape[1]: df_raw = df_sep
            except: pass
    # Strip whitespace from all column names
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw.loc[:, [c for c in df_raw.columns if c]]

    if df_raw.empty:
        return "Erreur: le fichier AFD est vide ou les colonnes sont illisibles."

    if not target_col:
        for candidate in ('Position', 'Pos', 'Target', 'Class', 'Classe'):
            resolved = resolve_column(candidate, df_raw.columns)
            if resolved:
                target_col = resolved
                break
        if not target_col:
            return "Erreur: veuillez choisir une variable cible pour l'AFD."
    
    # Case-insensitive lookup for target column
    target_col_actual = resolve_column(target_col, df_raw.columns)
    if not target_col_actual:
        return f"Erreur: Colonne cible '{target_col}' non trouvée dans les données. Colonnes disponibles: {list(df_raw.columns)}"
    target_col = target_col_actual
        
    # Normalize column names to find Player and Pos regardless of case
    col_map = {normalize_key(c): c for c in df_raw.columns}
    
    player_col = col_map.get('player')
    image_col = col_map.get('image_url')

    resolved_feature_cols = []
    for col in feature_cols:
        resolved = resolve_column(col, df_raw.columns)
        if resolved and resolved not in resolved_feature_cols:
            resolved_feature_cols.append(resolved)
    feature_cols = resolved_feature_cols
    
    # Exclude metadata and target from feature_cols
    exclude_cols = {normalize_key(target_col_actual)}
    if player_col:
        exclude_cols.add(normalize_key(player_col))
    if image_col:
        exclude_cols.add(normalize_key(image_col))

    if not feature_cols:
        feature_cols = [c for c in df_raw.columns if normalize_key(c) not in exclude_cols]
        
    cleaned_feature_cols = [c for c in feature_cols if normalize_key(c) not in exclude_cols]
    
    # Filter only numeric feature columns by checking their conversion viability
    valid_feature_cols = []
    for col in cleaned_feature_cols:
        if col in df_raw.columns:
            converted = coerce_numeric_series(df_raw[col])
            if converted.notna().sum() > 0.5 * len(converted):
                df_raw[col] = converted
                valid_feature_cols.append(col)

    if not valid_feature_cols:
        return "Erreur: aucune variable numérique valide n'a été trouvée pour l'AFD. Sélectionnez au moins une colonne contenant des nombres."
                
    extra_cols = []
    if player_col: extra_cols.append(player_col)
    if image_col: extra_cols.append(image_col)
        
    cols_to_use = [target_col_actual] + valid_feature_cols + extra_cols
    # Ensure all columns exist in df_raw
    cols_to_use = [c for c in cols_to_use if c in df_raw.columns]
    
    # Drop rows ONLY if target or valid features are missing
    essential_cols = [target_col_actual] + valid_feature_cols
    df = df_raw[cols_to_use].dropna(subset=essential_cols).copy()
    
    if df.empty:
        return "Erreur: Aucune donnée valide trouvée pour l'analyse après nettoyage. Assurez-vous d'avoir sélectionné des variables numériques."
        
    # Set Player as index if exists, ensuring 1D
    if player_col and player_col in df.columns:
        if isinstance(df[player_col], pd.DataFrame):
            temp_idx = df[player_col].iloc[:, 0]
            df.index = temp_idx
            df = df.drop(columns=[player_col])
        else:
            df = df.set_index(player_col)
        # Update target_col name to the actual one
        target_col = target_col_actual
        
        # Make index unique to avoid duplicate label issues in pandas operations
        new_index = []
        seen = {}
        for item in df.index:
            item_str = str(item)
            if item_str in seen:
                seen[item_str] += 1
                new_index.append(f"{item_str} ({seen[item_str]})")
            else:
                seen[item_str] = 1
                new_index.append(item_str)
        df.index = new_index
        
    # Position Mapping (Custom 3-class grouping for 'Pos' or 'Position' columns)
    if target_col.lower() in ('pos', 'position'):
        def map_position(pos):
            pos = str(pos).strip()
            pos_up = pos.upper()
            # Handle raw FBref codes: GK, DF, MF, FW (or combos like "MF,FW")
            first = pos_up.split(',')[0].strip()
            if first in ('GK', 'DF') or 'GK' in first or 'DF' in first:
                return 'Defense'
            elif first == 'MF' or 'MF' in first:
                return 'Milieu'
            elif first == 'FW' or 'FW' in first:
                return 'Attaque'
            # Handle already-mapped French labels from the transformed CSV
            pos_low = pos.lower()
            if any(x in pos_low for x in ('gardien', 'defenseur', 'défenseur', 'defense', 'défense')):
                return 'Defense'
            elif any(x in pos_low for x in ('milieu',)):
                return 'Milieu'
            elif any(x in pos_low for x in ('attaquant', 'attaque')):
                return 'Attaque'
            return 'Other'

        df[target_col] = df[target_col].apply(map_position)
        df = df[df[target_col] != 'Other']  # Remove unknown positions
        
    if df.empty:
        return "Erreur: Données vides après application de la cartographie des positions."

    # Filter out classes with only 1 member to ensure stratified splitting works reliably
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    if len(valid_classes) < len(class_counts):
        df = df[df[target_col].isin(valid_classes)].copy()
        
    if df.empty:
        return "Erreur: Données vides après filtrage des classes à membre unique."
        
    # Train/Test Split
    class_counts = df[target_col].value_counts()
    can_stratify = class_counts.min() >= 2
    if can_stratify:
        n_classes = len(class_counts)
        n_samples = len(df)
        min_test_size = n_classes / n_samples
        max_test_size = 1 - (n_classes / n_samples)
        if min_test_size <= max_test_size:
            test_size = min(max(test_size, min_test_size), max_test_size)
        else:
            return "Erreur: pas assez de lignes par classe pour créer un jeu train/test fiable."
    
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[target_col] if can_stratify else None,
            random_state=42
        )
    except ValueError as e:
        return f"Erreur lors du split train/test: {e}"
    
    # Pre-fit LabelEncoder on full dataset to prevent unseen classes issue in test split
    le = LabelEncoder()
    le.fit(df[target_col])
    
    from afd_logic import run_afd_calculation, calculate_afd_metrics
    
    # Run AFD on Training Set
    try:
        afd_train = run_afd_calculation(train_df, target_col, le=le)
    except ValueError as e:
        return f"Erreur lors de l'analyse : {str(e)}"
    
    # Project Test Set using same weights
    X_test = test_df[afd_train["feature_names"]].values
    X_test_proj = X_test @ afd_train["W_axes"]
    
    # Classification on Test Set (Simple Centroid-based Classifier)
    # Calculate centroids in projected space
    train_proj = afd_train["df_afd"][[c for c in afd_train["df_afd"].columns if c.startswith('LD')]].values
    train_y = afd_train["y_encoded"]
    
    centroids_proj = []
    for i in range(len(afd_train["classes"])):
        centroids_proj.append(np.mean(train_proj[train_y == i], axis=0))
    centroids_proj = np.array(centroids_proj)
    # Predict by nearest centroid
    preds = []
    for i in range(len(X_test_proj)):
        dists = np.linalg.norm(centroids_proj - X_test_proj[i], axis=1)
        preds.append(np.argmin(dists))

    # Metrics
    try:
        metrics = calculate_afd_metrics(afd_train["le"].transform(test_df[target_col]), preds, classes=afd_train["classes"])
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("AFD METRICS ERROR:", tb)
        return f"Erreur lors du calcul des métriques AFD: {e}<br><pre>{tb}</pre>"

    def matrix_html(values, index=None, columns=None, decimals=4):
        try:
            return pd.DataFrame(values, index=index, columns=columns).round(decimals).to_html(classes='table', border=0)
        except Exception as e:
            return f"<p class=\'text-secondary\'>[Erreur affichage matrice: {e}]</p>"

    try:
        ld_cols = [c for c in afd_train["df_afd"].columns if c.startswith('LD')]
        eig_vals_all = np.array(afd_train["eig_vals_all"], dtype=float)
        positive_eig_vals = np.clip(eig_vals_all, 0, None)
        eig_total = positive_eig_vals.sum()
        eig_percent = (positive_eig_vals / eig_total * 100).tolist() if eig_total > 0 else [0 for _ in eig_vals_all]
        selected_eig_percent = eig_percent[:len(ld_cols)]
        cumulative_quality = float(sum(selected_eig_percent))

        # Build the AFD plot explicitly so points stay visible after template/layout changes.
        afd_res_df = afd_train["df_afd"].copy()
        has_ld2 = "LD2" in afd_res_df.columns
        if has_ld2:
            y_values = afd_res_df["LD2"].astype(float)
            y_title = f"Axe 2 ({selected_eig_percent[1]:.2f}%)" if len(selected_eig_percent) > 1 else "Axe 2"
        else:
            y_values = pd.Series(np.zeros(len(afd_res_df)), index=afd_res_df.index)
            y_title = "Projection 1D"

        plot_df = pd.DataFrame({
            "LD1": afd_res_df["LD1"].astype(float),
            "LD2": y_values,
            "Target": afd_res_df["Target"].astype(str),
            "Individu": afd_res_df.index.astype(str)
        })
        palette = px.colors.qualitative.Set2 + px.colors.qualitative.Bold
        fig_afd = go.Figure()
        for i, (target, group) in enumerate(plot_df.groupby("Target", sort=True)):
            fig_afd.add_trace(go.Scatter(
                x=group["LD1"].astype(float).tolist(),
                y=group["LD2"].astype(float).tolist(),
                mode="markers",
                name=str(target),
                text=group["Individu"].astype(str).tolist(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Classe: {target}<br>"
                    "LD1: %{x:.4f}<br>"
                    "LD2: %{y:.4f}<extra></extra>"
                ),
                marker={
                    "size": 12,
                    "color": palette[i % len(palette)],
                    "opacity": 0.92,
                    "line": {"width": 1.5, "color": "#ffffff"}
                }
            ))

        if not has_ld2:
            fig_afd.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.35)")

        fig_afd.update_layout(
            title={"text": "Analyse Factorielle Discriminante", "x": 0.02, "xanchor": "left"},
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font={"color": "#f8fafc"},
            legend_title_text=target_col,
            margin={"l": 70, "r": 30, "t": 70, "b": 70},
            xaxis_title=f"Axe 1 ({selected_eig_percent[0]:.2f}%)" if selected_eig_percent else "Axe 1",
            yaxis_title=y_title
        )

        afd_steps = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(afd_train["feature_names"]),
            "n_classes": len(afd_train["classes"]),
            "q_axes": len(ld_cols),
            "class_counts": afd_train["class_counts"],
            "X_html": afd_train["X_df"].head(100).round(4).to_html(classes='table', border=0),
            "y_html": afd_train["y_series"].head(100).to_frame().to_html(classes='table', border=0),
            "global_mean_html": afd_train["global_mean"].to_frame("Centre global").T.round(4).to_html(classes='table', border=0),
            "class_means_html": afd_train["class_means"].round(4).to_html(classes='table', border=0),
            "S_W_html": matrix_html(afd_train["S_W"], afd_train["feature_names"], afd_train["feature_names"]),
            "S_B_html": matrix_html(afd_train["S_B"], afd_train["feature_names"], afd_train["feature_names"]),
            "S_W_reg_html": matrix_html(afd_train["S_W_reg"], afd_train["feature_names"], afd_train["feature_names"]),
            "W_inv_html": matrix_html(afd_train["W_inv"], afd_train["feature_names"], afd_train["feature_names"]),
            "fisher_matrix_html": matrix_html(afd_train["fisher_matrix"], afd_train["feature_names"], afd_train["feature_names"]),
            "eig_vals_all_html": pd.DataFrame({
                "Axe": [f"LD{i+1}" for i in range(len(afd_train["eig_vals_all"]))],
                "Valeur propre": afd_train["eig_vals_all"],
                "Pouvoir discriminant (%)": eig_percent
            }).round(6).to_html(classes='table', border=0, index=False),
            "axes_html": matrix_html(afd_train["W_axes"], afd_train["feature_names"], ld_cols, decimals=6),
            "projection_html": afd_train["df_afd"].head(100).round(4).to_html(classes='table', border=0),
            "eig_percent": eig_percent,
            "selected_eig_percent": selected_eig_percent,
            "quality_value": cumulative_quality
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("AFD STEPS BUILD ERROR:", tb)
        return f"Erreur lors de la construction des etapes AFD: {e}"

    try:
        return render_template("results_afd.html",
                               metrics=metrics,
                               afd_plot=fig_afd.to_json(),
                               classes=afd_train["classes"],
                               target_col=target_col,
                               initial_df=df.head(100).to_html(classes='table', border=0),
                               S_W=afd_train["S_W"],
                               S_B=afd_train["S_B"],
                               eig_vals=afd_train["eig_vals"],
                               eig_vecs=afd_train["eig_vecs"],
                               feature_names=afd_train["feature_names"],
                               afd_steps=afd_steps)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("AFD RENDER ERROR:", tb)
        return f"Erreur lors du rendu du template AFD: {e}"


@app.route("/get_interpretation", methods=["POST"])
def get_interpretation():
    data = request.json
    try:
        eig_percent = np.array(data['eig_percent'])
        coord_var = pd.DataFrame(data['coord_var'])
        C_vals = pd.DataFrame(data['C'])
        clusters = data.get('clusters', [])
        api_key = os.getenv('GROQ_API_KEY')
        interpretation = generate_llm_interpretation(eig_percent, coord_var, C_vals, clusters, api_key)
        return jsonify({"interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    chat_history = data.get('history', [])
    interpretation_data = data.get('interpretationData', {})
    user_message = data.get('message', '')
    api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        return jsonify({"error": "Cle API Groq manquante."}), 400

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        context_str = json.dumps({
            "eig_percent": interpretation_data.get("eig_percent"),
            "coord_var": interpretation_data.get("coord_var")
        }, ensure_ascii=False)

        system_prompt = f"""
Tu es un data scientist expert en Analyse en Composantes Principales (ACP) et un assistant de ce projet.
Reponds aux questions de l'utilisateur de maniere concise, professionnelle, et en francais.
Voici le contexte partiel des resultats actuels (valeurs propres en pourcentages, et coordonnees des variables sur les axes):
{context_str}
Ne donne ces informations brutes que si elles sont utiles pour repondre a la question de l'utilisateur.
Formate toujours ta reponse en markdown (utilisation de **gras**, de listes, etc. si necessaire).
"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        model_names = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        last_error = ""
        for model_name in model_names:
            try:
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                return jsonify({"response": chat_completion.choices[0].message.content})
            except Exception as e:
                last_error = str(e)
                if "400" in last_error or "model_decommissioned" in last_error or "404" in last_error:
                    continue
                else:
                    break
        return jsonify({"error": f"Erreur avec les modeles Groq: {last_error}"}), 500

    except Exception as e:
        print(f"DEBUG ERROR GROQ CHAT: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/launch_tableau", methods=["POST"])
def launch_tableau():
    try:
        data = request.json
        # We need the actual data to generate the Hyper file
        # The frontend doesn't send the data, but we can retrieve it or use the interpretation_data
        # Actually, it's better if we generate the files during run_pca and just launch them here
        # But wait, we want a 'one click' that always uses the current results.
        
        # Look for the last generated exports/analyse_acp.twbx
        export_dir = os.path.join(app.root_path, 'exports')
        twbx_path = os.path.join(export_dir, 'analyse_acp.twbx')
        
        if not os.path.exists(twbx_path):
            return jsonify({"error": "Fichier .twbx non trouvé. Relancez une analyse d'abord."}), 404
            
        # Launch Tableau
        if os.name == "nt": # Windows
            os.startfile(os.path.abspath(twbx_path))
        else:
            import subprocess
            subprocess.run(["open", twbx_path])
            
        return jsonify({"success": True, "path": twbx_path})
    except Exception as e:
        print(f"DEBUG LAUNCH TABLEAU ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================
# SCRAPING ROUTES
# ============================================================

LEAGUES_CONFIG = {
    "premier_league": {"name": "Premier League", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "ligue1":         {"name": "Ligue 1",        "flag": "🇫🇷"},
    "laliga":         {"name": "La Liga",        "flag": "🇪🇸"},
    "bundesliga":     {"name": "Bundesliga",     "flag": "🇩🇪"},
    "seriea":         {"name": "Serie A",        "flag": "🇮🇹"},
}

_scrape_state = {"running": False, "log": [], "output_file": None, "error": None}


@app.route("/scrape_page")
def scrape_page():
    return render_template("scrape.html", leagues=LEAGUES_CONFIG)


@app.route("/start_scrape", methods=["POST"])
def start_scrape():
    global _scrape_state
    if _scrape_state["running"]:
        return jsonify({"error": "Un scraping est déjà en cours."}), 409

    data = request.json
    league_key = data.get("league")
    step = data.get("step", "all")
    season = data.get("season", "")

    if league_key not in LEAGUES_CONFIG:
        return jsonify({"error": "Ligue invalide."}), 400

    _scrape_state = {"running": True, "log": [], "output_file": None, "error": None}

    def run():
        global _scrape_state
        try:
            import sys
            sys.path.insert(0, app.root_path)
            from scrape_league import scrape_stats, scrape_photos, clean_csv, LEAGUES

            output_dir = app.root_path

            # log_fn pipes scraper print() messages into the status endpoint
            def log_fn(msg):
                _scrape_state["log"].append(str(msg))

            if step in ("stats", "all"):
                log_fn(f"Scraping stats: {LEAGUES[league_key]['name']} (Saison: {season if season else 'Actuelle'})...")
                scrape_stats(league_key, season, output_dir, log_fn=log_fn)
                log_fn("Stats telecharges avec succes.")

            if step in ("photos", "all"):
                log_fn("Scraping photos des joueurs...")
                scrape_photos(league_key, output_dir, log_fn=log_fn)
                log_fn("Photos recuperees avec succes.")

            if step in ("clean", "all"):
                log_fn("Nettoyage du fichier CSV...")
                clean_path = clean_csv(league_key, output_dir, log_fn=log_fn)
                if clean_path:
                    _scrape_state["output_file"] = os.path.basename(clean_path)
                    _scrape_state["log"].append(f"Termine ! Fichier pret : {os.path.basename(clean_path)}")

        except Exception as e:
            _scrape_state["error"] = str(e)
            _scrape_state["log"].append(f"Erreur: {e}")
        finally:
            _scrape_state["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})



@app.route("/scrape_status")
def scrape_status():
    return jsonify({
        "running": _scrape_state["running"],
        "log": _scrape_state["log"],
        "output_file": _scrape_state["output_file"],
        "error": _scrape_state["error"]
    })


@app.route("/download_scraped/<filename>")
def download_scraped(filename):
    safe = os.path.basename(filename)
    path = os.path.join(app.root_path, safe)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Fichier non trouvé", 404


def create_hyper_file(dfs_dict, hyper_path):
    """
    Converts a dictionary of {table_name: df} to a single Tableau .hyper file.
    """
    with HyperProcess(Telemetry.SEND_USAGE_DATA_TO_TABLEAU, 'PCA_App') as hyper:
        with Connection(hyper.endpoint, hyper_path, CreateMode.CREATE_AND_REPLACE) as connection:
            connection.catalog.create_schema('Extract')
            
            for table_name, df in dfs_dict.items():
                columns = []
                for col in df.columns:
                    if col in ["Individu", "Name", "Type", "Image_URL"]:
                        columns.append(TableDefinition.Column(col, SqlType.text()))
                    elif col == "Cluster" or col == "Path":
                        columns.append(TableDefinition.Column(col, SqlType.int()))
                    else:
                        columns.append(TableDefinition.Column(col, SqlType.double()))
                
                table_def = TableDefinition(
                    table_name=TableName('Extract', table_name),
                    columns=columns
                )
                connection.catalog.create_table(table_def)
                
                with Inserter(connection, table_def) as inserter:
                    clean_df = df.where(pd.notnull(df), None)
                    for _, row in clean_df.iterrows():
                        inserter.add_row(list(row))
                    inserter.execute()

def get_packaged_xml(hyper_filename):
    """
    Generates a PRO Dashboard XML with 2 Sheets (Individus + Circle).
    (Dark mode styling removed due to Tableau strict XML parsing issues)
    """
    xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook source-build='2025.3.0' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <datasources>
    <datasource caption='Table Individus' inline='true' name='ds_ind' version='18.1'>
      <connection class='hyper' dbname='Data/Extracts/{hyper_filename}' server='' table='[Extract].[Individus]' />
      <column datatype='string' name='[Individu]' role='dimension' type='nominal' />
      <column datatype='real' name='[C1]' role='measure' type='quantitative' />
      <column datatype='real' name='[C2]' role='measure' type='quantitative' />
      <column datatype='integer' name='[Cluster]' role='measure' type='nominal' />
      <column datatype='string' name='[Image_URL]' role='dimension' type='nominal' />
    </datasource>
    <datasource caption='Table Variables' inline='true' name='ds_var' version='18.1'>
      <connection class='hyper' dbname='Data/Extracts/{hyper_filename}' server='' table='[Extract].[Variables]' />
      <column datatype='string' name='[Name]' role='dimension' type='nominal' />
      <column datatype='real' name='[C1]' role='measure' type='quantitative' />
      <column datatype='real' name='[C2]' role='measure' type='quantitative' />
      <column datatype='integer' name='[Path]' role='measure' type='quantitative' />
      <column datatype='string' name='[Type]' role='dimension' type='nominal' />
    </datasource>
  </datasources>

  <worksheets>
    <!-- SHEET 1: INDIVIDUS -->
    <worksheet name='Nuage Individus'>
      <table>
        <view>
          <datasources>
            <datasource name='ds_ind' />
          </datasources>
          <datasource-dependencies datasource='ds_ind'>
            <column-instance column='[C1]' derivation='None' name='[none:C1:qk]' pivot='key' type='quantitative' />
            <column-instance column='[C2]' derivation='None' name='[none:C2:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Cluster]' derivation='None' name='[none:Cluster:nk]' pivot='key' type='nominal' />
            <column-instance column='[Individu]' derivation='None' name='[none:Individu:nk]' pivot='key' type='nominal' />
            <column-instance column='[Image_URL]' derivation='None' name='[none:Image_URL:nk]' pivot='key' type='nominal' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view><breakdown value='auto' /></view>
            <mark class='Circle' />
            <encodings>
              <color column='[ds_ind].[none:Cluster:nk]' />
              <text column='[ds_ind].[none:Individu:nk]' />
              <tooltip column='[ds_ind].[none:Image_URL:nk]' />
            </encodings>
            <style><style-rule element='mark'><format attr='mark-labels-show' value='true' /></style-rule></style>
          </pane>
        </panes>
        <rows>[ds_ind].[none:C2:qk]</rows>
        <cols>[ds_ind].[none:C1:qk]</cols>
      </table>
    </worksheet>

    <!-- SHEET 2: CERCLE VARIABLES -->
    <worksheet name='Cercle Corrélation'>
      <table>
        <view>
          <datasources><datasource name='ds_var' /></datasources>
          <datasource-dependencies datasource='ds_var'>
            <column-instance column='[C1]' derivation='None' name='[none:C1:qk]' pivot='key' type='quantitative' />
            <column-instance column='[C2]' derivation='None' name='[none:C2:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Name]' derivation='None' name='[none:Name:nk]' pivot='key' type='nominal' />
            <column-instance column='[Path]' derivation='None' name='[none:Path:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Type]' derivation='None' name='[none:Type:nk]' pivot='key' type='nominal' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view><breakdown value='auto' /></view>
            <mark class='Line' />
            <encodings>
                <color column='[ds_var].[none:Type:nk]' />
                <path column='[ds_var].[none:Path:qk]' />
                <lod column='[ds_var].[none:Name:nk]' />
                <text column='[ds_var].[none:Name:nk]' />
            </encodings>
            <style><style-rule element='mark'><format attr='mark-labels-show' value='true' /></style-rule></style>
          </pane>
        </panes>
        <rows>[ds_var].[none:C2:qk]</rows>
        <cols>[ds_var].[none:C1:qk]</cols>
      </table>
    </worksheet>
  </worksheets>

  <dashboards>
    <dashboard name='Dashboard Analyse'>
      <style />
      <size maxpx='1400' minpx='1000' preset_size_prop='1400x800' />
      <zones>
        <zone type='layout-basic' force-tiled-v-ab-order='true'>
          <zone type='layout-flow' orient='horiz'>
            <zone name='Nuage Individus' type='worksheet' />
            <zone name='Cercle Corrélation' type='worksheet' />
          </zone>
        </zone>
      </zones>
    </dashboard>
  </dashboards>

  <windows orientation='horizontal'>
    <window class='worksheet' name='Nuage Individus' />
    <window class='worksheet' name='Cercle Corrélation' />
    <window class='dashboard' maximized='true' name='Dashboard Analyse' />
  </windows>
</workbook>"""
    return xml

def generate_llm_interpretation(eig_percent, coord_var, F, clusters, api_key=None):
    """
    Generates a statistical PCA interpretation using Llama-3 via Groq API.
    Returns HTML text.
    """
    try:
        if not api_key:
            return "<p><i class='fas fa-exclamation-triangle'></i> Clé API Groq manquante.</p>"

        client = Groq(api_key=api_key)

        n_ind = len(F)
        n_var = len(coord_var)
        headers = list(coord_var.index)
        eigenvals = eig_percent
        cumulative = np.cumsum(eig_percent)
        corr_text = coord_var.round(3).to_dict(orient="index")
        top_N = min(n_ind, 20)
        ind_text = F.head(top_N).round(3).to_dict(orient="index")
        method = "ACP"

        prompt = f"""
Tu es un expert en analyse de données (ACP). Interprète les résultats suivants :
- Méthode ACP: {method}
- Individus: {n_ind}, Variables: {n_var}
- Variables: {', '.join(headers)}
- Valeurs propres: {[round(v, 3) for v in eigenvals]}
- Variance cumulée: {[round(v, 1) for v in cumulative]}
- Corrélations variables-axes: {corr_text}

Coordonnées des individus (Top {top_N}) sur les 2 premiers axes:
{ind_text}

IMPORTANT : Structure ta réponse EXACTEMENT avec les balises suivantes :

[[AXES]]
(Interprète ici F1, F2, etc. en rapport avec les variables et leur sens physique)

[[PARTITIONS]]
(Propose ici obligatoirement des partitions en 2 classes ET en 3 classes.
Pour CHAQUE partition, indique :
1. Le nom de la classe
2. Les caractéristiques (pourquoi ils sont ensemble)
3. La liste des individus appartenant à cette classe parmi ceux fournis ci-dessus)

[[ATYPIQUES]]
(Identifie ici les joueurs "atypiques", c'est-à-dire ceux qui s'écartent significativement du centre du nuage de points sur les axes F1 et F2, et explique pourquoi ils sont considérés comme atypiques)

[[CONCLUSION]]
(Donne ici une conclusion sur la structure globale des données et le lien entre variables et individus)

RÉPONDS EN FRANÇAIS.
"""

        # Modèles Groq mis à jour (le 70b-8192 a été retiré)
        model_names = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        last_error = ""

        for model_name in model_names:
            try:
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_text = chat_completion.choices[0].message.content
                # Convert the custom text tags to styled HTML for the frontend
                formatted_html = raw_text.replace('\n', '<br>')
                formatted_html = formatted_html.replace('[[AXES]]', '<h3 style="color:var(--accent-color); margin-top:1.5rem;"><i class="fas fa-arrows-alt"></i> Axes Factoriels</h3>')
                formatted_html = formatted_html.replace('[[PARTITIONS]]', '<h3 style="color:var(--neon-purple); margin-top:1.5rem;"><i class="fas fa-layer-group"></i> Partitions Proposées</h3>')
                formatted_html = formatted_html.replace('[[ATYPIQUES]]', '<h3 style="color:var(--neon-cyan); margin-top:1.5rem;"><i class="fas fa-star"></i> Individus Atypiques</h3>')
                formatted_html = formatted_html.replace('[[CONCLUSION]]', '<h3 style="color:var(--success-color); margin-top:1.5rem;"><i class="fas fa-flag-checkered"></i> Conclusion</h3>')
                
                # Simple fix for markdown bolding if the LLM uses it
                formatted_html = formatted_html.replace('**', '<b>') 
                
                return formatted_html
            except Exception as e:
                last_error = str(e)
                # Si le modèle n'existe plus ou est surchargé, on tente le suivant
                if "400" in last_error or "model_decommissioned" in last_error or "404" in last_error:
                    continue
                else: break

        return f"<p><i class='fas fa-bug'></i> Aucun modèle Groq n'a pu répondre : {last_error}</p>"
        
    except Exception as e:
        print(f"DEBUG ERROR GROQ CRITICAL: {str(e)}")
        return f"<p><i class='fas fa-bug'></i> Erreur critique Groq : {str(e)}</p>"

def run_pca(df, acp_type, data_type, criterion, image_urls=None, api_key=None):
    n, p = df.shape
    
    if acp_type == 'norme':
        stds = df.std(ddof=0).replace(0, 1)
        Z = (df - df.mean()) / stds
        X_centered = df - df.mean()
        data_matrix = Z
        matrix_label = "Matrice de Corrélation (R)"
        matrix_formula = "$R = \\frac{1}{n} Z^T Z$"
        eig_matrix_symbol = "R"
        component_formula = "$C = Z \\cdot V$"
        standardization_label = "Matrice Z-score"
        standardization_formula = "$Z = \\frac{X - \\bar{X}}{\\sigma}$"
    else:
        X_centered = df - df.mean()
        Z = X_centered
        data_matrix = X_centered
        matrix_label = "Matrice de Covariance (V)"
        matrix_formula = "$V = \\frac{1}{n} X_c^T X_c$"
        eig_matrix_symbol = "V"
        component_formula = "$C = X_c \\cdot V$"
        standardization_label = "Matrice Centrée"
        standardization_formula = "$X_c = X - \\bar{X}$"

    if acp_type == 'norme':
        V = (data_matrix.T @ data_matrix) / n
        metric_matrix = None
    else:
        V = (data_matrix.T @ data_matrix) / n
        metric_matrix = None
        if data_type == 'heterogene':
            variances = df.var(ddof=0)
            if (variances == 0).any(): variances = variances.replace(0, 1e-10)
            metric_matrix = np.diag(1.0 / variances.values)
            matrix_label = "Matrice de Covariance (V)"
            matrix_formula = "$V = \\frac{1}{n} X_c^T X_c \\text{ (avec métrique } \\frac{1}{\\text{Var}} \\text{ pour calcul)}$"
            eig_matrix_symbol = "VM"
            component_formula = "$C = X_c \\cdot M \\cdot V$"

    if acp_type == 'non_norme' and data_type == 'heterogene' and metric_matrix is not None:
        VM = V @ metric_matrix
        eig_vals, eig_vecs_VM = np.linalg.eig(VM)
        eig_vals = eig_vals.real
        eig_vecs_VM = eig_vecs_VM.real
        
        # M-Normalization: U^T M U = 1
        # eig_vecs_VM are columns U_k. We need U_k^T M U_k = 1.
        for k in range(eig_vecs_VM.shape[1]):
            u_k = eig_vecs_VM[:, k]
            # norm_sq = u_k.T @ M @ u_k
            norm_sq = u_k @ metric_matrix @ u_k
            if norm_sq > 0:
                eig_vecs_VM[:, k] = u_k / np.sqrt(norm_sq)
        
        eig_vecs = eig_vecs_VM
        if np.isnan(eig_vecs).any(): eig_vecs = np.nan_to_num(eig_vecs)
        display_matrix = pd.DataFrame(V, index=df.columns, columns=df.columns)
    else:
        eig_vals, eig_vecs = np.linalg.eigh(V)
        display_matrix = pd.DataFrame(V, index=df.columns, columns=df.columns)
        
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    total_inertia = np.sum(eig_vals)
    Q = []
    for i in range(len(eig_vals)):
        q = np.sum(eig_vals[:i+1]) / total_inertia if total_inertia > 0 else 0
        Q.append(float(round(q*100, 2)))
        
    if criterion == 'kaiser':
        mean_l = np.mean(eig_vals)
        nb_axes = np.sum(eig_vals > mean_l)
        crit_text = f"Kaiser (> {mean_l:.4f})"
    else:
        nb_axes = 0
        for i, q in enumerate(Q):
            if q >= 80: nb_axes = i + 1; break
        if nb_axes == 0: nb_axes = len(eig_vals)
        crit_text = "Qualité >= 80%"
        
    nb_axes_for_plot = nb_axes
    if nb_axes_for_plot == 0 and len(eig_vals) >= 1:
        nb_axes_for_plot = 1
        
    V_axes = eig_vecs[:, :nb_axes_for_plot]
    
    if acp_type == 'non_norme' and data_type == 'heterogene' and metric_matrix is not None:
        # F = X @ M @ U
        F_vals = data_matrix @ metric_matrix @ V_axes
    else:
        F_vals = data_matrix.dot(V_axes)
        
    if isinstance(F_vals, pd.DataFrame): F_vals = F_vals.values
    F_vals = np.real(F_vals)
    F = pd.DataFrame(F_vals, index=df.index, columns=[f"C{i+1}" for i in range(nb_axes_for_plot)])

    coord_var = eig_vecs * np.sqrt(np.abs(eig_vals))
    df_coord_var = pd.DataFrame(coord_var[:, :nb_axes_for_plot], index=df.columns, columns=[f"C{i+1}" for i in range(nb_axes_for_plot)])
    
    if acp_type == 'non_norme':
        stds = df.std(ddof=0).replace(0, 1)
        df_coord_var = df_coord_var.div(stds, axis=0)

    plot_coords = df_coord_var

    eig_percent = (eig_vals / total_inertia) * 100

    # --- Calculations for COS2 and CTR ---
    X_vals = data_matrix.values
    if metric_matrix is not None:
        # Multiplier par la métrique (1/var) pour ACP non normée hétérogène
        weights = np.diag(metric_matrix)
        dist_sq = np.sum((X_vals ** 2) * weights, axis=1)
    else:
        dist_sq = np.sum(X_vals ** 2, axis=1)
        
    dist_sq[dist_sq == 0] = 1e-10 # Prevent division by zero
    
    F_sq = F_vals ** 2
    cos2_vals = F_sq / dist_sq[:, np.newaxis]
    df_cos2 = pd.DataFrame(cos2_vals, index=df.index, columns=[f"C{i+1}" for i in range(nb_axes_for_plot)])
    
    # --- Generate COS2 Cards instead of Table ---
    cos2_html = "<div class='grid-2' style='gap: 1rem;'>"
    max_cards = 50 # Prevent massive DOM size
    for i in range(min(n, max_cards)):
        for k in range(nb_axes_for_plot):
            C_val = F_vals[i, k]
            dist = dist_sq[i]
            c2 = cos2_vals[i, k]
            C_sq = C_val ** 2
            cos2_html += f"<div class='card' style='padding: 1rem; margin: 0;'><div class='formula' style='margin:0; font-size: 0.95rem;'>$\\cos^2(\\theta_{{{i+1}{k+1}}}) = \\frac{{({C_val:.2f})^2}}{{{dist:.2f}}} = \\frac{{{C_sq:.3f}}}{{{dist:.2f}}} = {c2:.3f}$</div></div>"
    cos2_html += "</div>"
    if n > max_cards:
        cos2_html += f"<p style='margin-top: 1rem; color: #94a3b8;'>* Affichage sous forme de cartes limité aux {max_cards} premiers individus pour préserver les performances.</p>"
    # -----------------------------------
    
    # Absolute Contributions (rho_ik) = (p_i * F^2) / lambda
    # Here p_i = 1/n
    ctr_vals = ( (1/n) * F_sq ) / eig_vals[:nb_axes_for_plot]
    df_ctr = pd.DataFrame(ctr_vals, index=df.index, columns=[f"C{i+1}" for i in range(nb_axes_for_plot)])
    
    # --- Generate CTR Cards instead of Table ---
    ctr_html = "<div class='grid-2' style='gap: 1rem;'>"
    max_cards = 50 # Prevent massive DOM size if there are hundreds of individuals
    for i in range(min(n, max_cards)):
        for k in range(nb_axes_for_plot):
            C_val = F_vals[i, k]
            lam = eig_vals[k]
            rho = ctr_vals[i, k]
            num_val = (1/n) * (C_val**2)
            ctr_html += f"<div class='card' style='padding: 1rem; margin: 0;'><div class='formula' style='margin:0; font-size: 0.95rem;'>$\\rho_{{{i+1}{k+1}}} = \\frac{{\\frac{{1}}{{{n}}} \\cdot ({C_val:.2f})^2}}{{{lam:.2f}}} = \\frac{{{num_val:.3f}}}{{{lam:.2f}}} = {rho:.3f}$</div></div>"
    ctr_html += "</div>"
    if n > max_cards:
        ctr_html += f"<p style='margin-top: 1rem; color: #94a3b8;'>* Affichage sous forme de cartes limité aux {max_cards} premiers individus pour préserver les performances.</p>"
    # -----------------------------------

    # K-Means Clustering on F coordinates
    try:
        n_clusters = min(3, len(df))
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        clusters = kmeans.fit_predict(F)
    except:
        clusters = np.zeros(len(df), dtype=int)

    # Data for Deferred AI Interpretation (Sanitized for JSON)
    # Sending up to the first 3 components (F1, F2, F3) and clusters
    interpretation_data = {
        "eig_percent": np.nan_to_num(eig_percent[:3]).tolist(),
        "coord_var": df_coord_var.iloc[:, :3].fillna(0).to_dict(),
        "C": F.iloc[:, :3].fillna(0).to_dict(),
        "clusters": clusters.tolist()
    }

    # Multi-Plot Generation
    plots_data = []
    
    # Ensure data is clean for rendering
    F = F.fillna(0)
    df_coord_var = df_coord_var.fillna(0)
    
    # Respect the number of axes retained
    axes_to_plot = nb_axes_for_plot
    axes_combinations = list(itertools.combinations(range(axes_to_plot), 2))
    
    # Colors for clusters (Matplotlib style blue if no clusters, or categorical)
    cluster_colors = ['#6366f1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
    clean_clusters = [int(c) if (not np.isnan(c) and not np.isinf(c)) else 0 for c in clusters]

    # Handle the case where we only have 1 axis (1D Plot)
    if not axes_combinations and axes_to_plot == 1:
        ax1 = 0
        
        # 1. Nuage des Individus (1D)
        fig_nuage = go.Figure()

        x_min, x_max = F.iloc[:, ax1].min(), F.iloc[:, ax1].max()
        dx = max(0.1, (x_max - x_min) * 0.15)
        dy = 0.5

        fig_nuage.add_trace(go.Scatter(
            x=F.iloc[:, ax1].tolist(),
            y=[0] * len(F),
            mode='markers+text',
            text=df.index.tolist(),
            textposition="bottom center",
            textfont={'family': "Inter, sans-serif", 'size': 11, 'color': "white"},
            marker={'size': 8, 'color': '#6366f1', 'line': {'width': 1, 'color': 'white'}, 'opacity': 0.85},
            hovertemplate="<b>%{text}</b><br>C1: %{x:.3f}<extra></extra>"
        ))

        images = []
        MAX_AVATARS = 100
        for j, player_name in enumerate(df.index.tolist()):
            if j >= MAX_AVATARS: break
            if image_urls is not None and j < len(image_urls) and pd.notna(image_urls.iloc[j]) and str(image_urls.iloc[j]).startswith('http'):
                img_url = image_urls.iloc[j]
            else:
                img_url = "https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y&s=128"
            images.append(dict(
                source=img_url, xref="x", yref="y",
                x=F.iloc[j, ax1], y=0,
                sizex=dx, sizey=dy,
                xanchor="center", yanchor="middle", layer="above"
            ))

        fig_nuage.update_layout(
            images=images, template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font={'color': 'white'},
            xaxis={'title': f"C1 ({round(eig_percent[ax1], 2)}%)", 'zeroline': True,
                   'zerolinecolor': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'showgrid': False, 'zeroline': True, 'zerolinecolor': 'white',
                   'showticklabels': False, 'range': [-1, 1]},
            margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
            hovermode='closest'
        )

        # 2. Cercle de Corrélation (1D)
        fig_cercle = go.Figure()
        fig_cercle.add_shape(
            type="circle", xref="x", yref="y",
            x0=-1, y0=-1, x1=1, y1=1,
            line={'color': "rgba(255,255,255,0.5)", 'width': 2, 'dash': "dash"}
        )

        for i, var_name in enumerate(df.columns):
            vx = float(df_coord_var.iloc[i, ax1])
            fig_cercle.add_annotation(
                x=vx, y=0, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2.5, arrowcolor="crimson",
                text=""
            )
            fig_cercle.add_annotation(
                x=vx, y=0,
                text=f"<b>{var_name}</b>",
                showarrow=False,
                xref="x", yref="y",
                font={'color': "white", 'size': 13, 'family': "Inter, sans-serif"},
                yshift=18 if vx >= 0 else -18
            )

        fig_cercle.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font={'color': 'white'},
            xaxis={'title': f"C1 ({round(eig_percent[ax1], 2)}%)", 'range': [-1.2, 1.2],
                   'zeroline': True, 'zerolinecolor': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'title': "", 'range': [-1.2, 1.2], 'scaleanchor': "x", 'scaleratio': 1,
                   'zeroline': True, 'zerolinecolor': 'white', 'showticklabels': False},
            showlegend=False, margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
        )

        plots_data.append({'ax1': 1, 'ax2': '', 'is_1d': True, 'nuage_json': fig_nuage.to_json(), 'cercle_json': fig_cercle.to_json()})



    for ax1, ax2 in axes_combinations:
        # 1. Nuage des Individus
        fig_nuage = go.Figure()
        
        # Calculate dynamic size for layout images
        x_min, x_max = F.iloc[:, ax1].min(), F.iloc[:, ax1].max()
        y_min, y_max = F.iloc[:, ax2].min(), F.iloc[:, ax2].max()
        dx = max(0.1, (x_max - x_min) * 0.15)
        dy = max(0.1, (y_max - y_min) * 0.15)

        # Add points
        fig_nuage.add_trace(go.Scatter(
            x=F.iloc[:, ax1].tolist(),
            y=F.iloc[:, ax2].tolist(),
            mode='markers+text',
            text=df.index.tolist(),
            textposition="bottom center",
            textfont={'family': "Inter, sans-serif", 'size': 12, 'color': "white"},
            marker={
                'size': 8,
                'color': '#6366f1',
                'line': {'width': 1, 'color': 'white'},
                'opacity': 0.85
            },
            hovertemplate="<b>%{text}</b><br>C" + str(ax1+1) + ": %{x:.2f}<br>C" + str(ax2+1) + ": %{y:.2f}<extra></extra>"
        ))
        
        import urllib.parse
        images = []
        MAX_AVATARS = 100
        for j, player_name in enumerate(df.index.tolist()):
            if j >= MAX_AVATARS: break
            
            if image_urls is not None and j < len(image_urls) and pd.notna(image_urls.iloc[j]) and str(image_urls.iloc[j]).startswith('http'):
                img_url = image_urls.iloc[j]
            else:
                # Fallback to generic silhouette profile picture (like Instagram default)
                img_url = "https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y&s=128"
                
            images.append(dict(
                source=img_url,
                xref="x", yref="y",
                x=F.iloc[j, ax1], y=F.iloc[j, ax2],
                sizex=dx, sizey=dy,
                xanchor="center", yanchor="middle",
                layer="above"
            ))

        # Styling for Individuals Cloud (Dark Theme)
        fig_nuage.update_layout(
            images=images,
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font={'color': 'white'},
            xaxis={'title': f"C{ax1+1} ({round(eig_percent[ax1], 2)}%)", 'zeroline': True, 'zerolinecolor': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'title': f"C{ax2+1} ({round(eig_percent[ax2], 2)}%)", 'zeroline': True, 'zerolinecolor': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
            margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
            hovermode='closest'
        )

        # 2. Cercle de Corrélation
        fig_cercle = go.Figure()
        
        # Add a real Circle SHAPE for maximum reliability
        fig_cercle.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-1, y0=-1, x1=1, y1=1,
            line={'color': "rgba(255,255,255,0.5)", 'width': 2, 'dash': "dash"}
        )
        
        # Add vectors matching the requested look
        for i, var_name in enumerate(df.columns):
            vx = float(df_coord_var.iloc[i, ax1])
            vy = float(df_coord_var.iloc[i, ax2])
            # The Arrow (Vector)
            fig_cercle.add_annotation(
                x=vx, y=vy, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2.5, arrowcolor="crimson",
                text=""
            )
            # The Text Label (placed at the tip)
            fig_cercle.add_annotation(
                x=vx, y=vy,
                text=f"<b>{var_name}</b>",
                showarrow=False,
                xref="x", yref="y",
                font={'color': "white", 'size': 13, 'family': "Inter, sans-serif"},
                yshift=15 if vy >= 0 else -15
            )

        fig_cercle.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font={'color': 'white'},
            xaxis={'title': f"C{ax1+1} ({round(eig_percent[ax1], 2)}%)", 'range': [-1.2, 1.2], 'zeroline': True, 'zerolinecolor': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
            yaxis={'title': f"C{ax2+1} ({round(eig_percent[ax2], 2)}%)", 'range': [-1.2, 1.2], 'scaleanchor': "x", 'scaleratio': 1, 'zeroline': True, 'zerolinecolor': 'white', 'gridcolor': 'rgba(255,255,255,0.1)'},
            showlegend=False,
            margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
        )
        
        plots_data.append({
            'ax1': ax1+1, 'ax2': ax2+1,
            'nuage_json': fig_nuage.to_json(),
            'cercle_json': fig_cercle.to_json()
        })
    

    result_dict = {
        'standardization': data_matrix.to_html(classes='table', border=0),
        'standardization_label': standardization_label,
        'standardization_formula': standardization_formula,
        'matrix': display_matrix.to_html(classes='table', border=0),
        'matrix_label': matrix_label,
        'matrix_formula': matrix_formula,
        'eig_vals': eig_vals.tolist(),
        'component_formula': component_formula,
        'C': F.round(4).to_html(classes='table', border=0),
        'coord_var': df_coord_var.round(4).to_html(classes='table', border=0),
        'cos2': cos2_html,
        'ctr': ctr_html,
        'eig_vals_df': pd.DataFrame({'Val': eig_vals}).to_html(classes='table', border=0),
        'eig_vecs': pd.DataFrame(eig_vecs, index=df.columns).to_html(classes='table', border=0),
        'Q_cumulative': Q,
        'nb_axes': nb_axes,
        'criterion_text': crit_text,
        'quality_value': Q[int(nb_axes)-1] if int(nb_axes) > 0 and len(Q) >= int(nb_axes) else 0,
        'plots': plots_data,
        'interpretation_data': interpretation_data,
        'component_formula': component_formula,
        'eig_matrix_symbol': eig_matrix_symbol
    }

    # --- Real Method: Packaged Workbook (.twbx) with Hyper ---
    try:
        export_dir = os.path.join(app.root_path, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        hyper_filename = "data_acp.hyper"
        hyper_path = os.path.join(export_dir, hyper_filename)
        
        # 1. Prepare Data for Hyper (Individuals)
        ind_export_df = F.copy()
        ind_export_df['Individu'] = ind_export_df.index
        ind_export_df['Cluster'] = clusters
        if image_urls is not None:
            ind_export_df['Image_URL'] = image_urls.values
        else:
            ind_export_df['Image_URL'] = "https://www.gravatar.com/avatar/00000000000000000000000000000000?s=128&d=mp"
        
        # 2. Prepare Data for Hyper (Variables/Circle)
        var_rows = []
        # Add circle points
        for i in range(101):
            angle = (i / 100) * 2 * np.pi
            var_rows.append({
                'Type': 'Cercle', 'Name': 'Cercle', 
                'C1': np.cos(angle), 'C2': np.sin(angle), 
                'Path': i
            })
        # Add vectors
        for i, var_name in enumerate(df.columns):
            vx = float(df_coord_var.iloc[i, 0])
            vy = float(df_coord_var.iloc[i, 1])
            var_rows.append({'Type': 'Vecteur', 'Name': var_name, 'C1': 0.0, 'C2': 0.0, 'Path': 0})
            var_rows.append({'Type': 'Vecteur', 'Name': var_name, 'C1': vx, 'C2': vy, 'Path': 1})
        
        var_export_df = pd.DataFrame(var_rows)

        # Create Hyper File with both tables
        create_hyper_file({
            "Individus": ind_export_df,
            "Variables": var_export_df
        }, hyper_path)
        
        # Create TWB (XML)
        twb_content = get_packaged_xml(hyper_filename)
        twb_path = os.path.join(export_dir, "analyse_acp.twb")
        with open(twb_path, "w", encoding="utf-8") as f:
            f.write(twb_content)
            
        # Create TWBX (ZIP)
        twbx_path = os.path.join(export_dir, "analyse_acp.twbx")
        with zipfile.ZipFile(twbx_path, 'w') as zipf:
            zipf.write(twb_path, "analyse_acp.twb")
            zipf.write(hyper_path, f"Data/Extracts/{hyper_filename}")
            
    except Exception as e:
        print(f"Error creating TWBX: {e}")

    return result_dict

if __name__ == "__main__":
    app.run(debug=True, port=5018)
