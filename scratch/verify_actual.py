import pandas as pd
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from afd_logic import run_afd_calculation

def verify():
    csv_file = r"c:\Users\DZ Laptops\Desktop\ACP\premier_league_players_cleaned.csv"
    df = pd.read_csv(csv_file)
    
    # Let's keep only a few numeric columns and the target
    feature_cols = ['Age', 'Playing Time_Min', 'Performance_Gls', 'Performance_Ast']
    target_col = 'Pos'
    
    # Mimic position mapping
    def map_position(pos):
        pos = str(pos).upper()
        if 'GK' in pos or 'DF' in pos:
            return 'Defender'
        elif 'MF' in pos:
            return 'Milieu'
        elif 'FW' in pos:
            return 'Attaque'
        return 'Other'
        
    df[target_col] = df[target_col].apply(map_position)
    df = df[df[target_col] != 'Other']
    
    df_to_calc = df[[target_col] + feature_cols].dropna()
    print("Shape before calculation:", df_to_calc.shape)
    
    # Run calculation
    res = run_afd_calculation(df_to_calc, target_col)
    print("Calculation successful!")
    print("Eigenvalues:", res['eig_vals'])
    print("Classes:", res['classes'])
    print("Feature names:", res['feature_names'])

if __name__ == '__main__':
    verify()
