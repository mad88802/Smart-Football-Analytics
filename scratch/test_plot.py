import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afd_logic import run_afd_calculation
import plotly.express as px

csv_file = r"c:\Users\DZ Laptops\Desktop\ACP\premier_league_players_cleaned.csv"
df_raw = pd.read_csv(csv_file)
df_raw.columns = df_raw.columns.str.strip()

# Create a 2-class subset
df = df_raw[df_raw['Pos'].isin(['GK', 'FW'])].copy()

# Let's select some features dynamically
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c not in ['Player', 'Nation', 'Pos', 'Squad', 'Image_URL', 'Target']][:5]
df = df[['Pos'] + features].dropna()

le = LabelEncoder()
y_encoded = le.fit_transform(df['Pos'])

# Run AFD calculation
afd_train = run_afd_calculation(df, 'Pos', le=le)
afd_res_df = afd_train["df_afd"]

print("Columns in df_afd:", afd_res_df.columns.tolist())
print("Head of df_afd:\n", afd_res_df.head())

# Generate Plotly plot as in app.py
fig_afd = px.scatter(
    afd_res_df, x="LD1", y="LD2" if "LD2" in afd_res_df.columns else "LD1",
    color="Target", title="Analyse Factorielle Discriminante - Espace Réduit",
    template="plotly_dark",
    hover_name=afd_res_df.index,
    labels={"LD1": f"Axe 1 ({round(afd_train['eig_vals'][0], 2)}%)",
            "LD2": f"Axe 2 ({round(afd_train['eig_vals'][1], 2)}%)" if len(afd_train['eig_vals']) > 1 else ""}
)

fig_json = json.loads(fig_afd.to_json())
print("\nNumber of traces in plot:", len(fig_json['data']))
if len(fig_json['data']) > 0:
    for i, trace in enumerate(fig_json['data']):
        print(f"Trace {i} (name={trace.get('name')}, type={trace.get('type')}):")
        x_val = trace.get('x')
        y_val = trace.get('y')
        print(f"  x type: {type(x_val)}, y type: {type(y_val)}")
        if isinstance(x_val, list):
            print("  x (first 5):", x_val[:5])
        else:
            print("  x:", x_val)
        if isinstance(y_val, list):
            print("  y (first 5):", y_val[:5])
        else:
            print("  y:", y_val)
