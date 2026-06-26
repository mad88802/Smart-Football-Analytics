import numpy as np
import pandas as pd


def run_afc_calculation(df):
    """
    Performs full Correspondence Analysis (AFC / ACM contingency) on a
    contingency table.

    Parameters
    ----------
    df : pd.DataFrame
        First column = row labels (str).
        Remaining columns = numeric counts.

    Returns
    -------
    dict with all intermediate steps and final coordinates.
    """
    # ------------------------------------------------------------------ #
    # 1. Raw contingency table
    # ------------------------------------------------------------------ #
    row_labels = df.iloc[:, 0].astype(str).tolist()
    col_labels = df.columns[1:].tolist()
    X = df.iloc[:, 1:].values.astype(float)

    n_rows, n_cols = X.shape

    # ------------------------------------------------------------------ #
    # 2. Grand total & relative frequencies (P matrix)
    # ------------------------------------------------------------------ #
    n = np.sum(X)
    if n == 0:
        raise ValueError("Le tableau de contingence est vide (total = 0).")

    fij = X / n  # relative frequencies

    # ------------------------------------------------------------------ #
    # 3. Row & column masses
    # ------------------------------------------------------------------ #
    fi = np.sum(fij, axis=1)   # (n_rows,)
    fj = np.sum(fij, axis=0)   # (n_cols,)

    # Guard against zero masses
    fi_safe = np.where(fi == 0, 1e-10, fi)
    fj_safe = np.where(fj == 0, 1e-10, fj)

    # ------------------------------------------------------------------ #
    # 4. Diagonal weight matrices (inverse square-root)
    # ------------------------------------------------------------------ #
    Di_inv_sqrt = np.diag(1.0 / np.sqrt(fi_safe))
    Dj_inv_sqrt = np.diag(1.0 / np.sqrt(fj_safe))

    # ------------------------------------------------------------------ #
    # 5. Centered / standardised residual matrix S
    # ------------------------------------------------------------------ #
    expected = np.outer(fi_safe, fj_safe)
    S = Di_inv_sqrt @ (fij - expected) @ Dj_inv_sqrt

    # ------------------------------------------------------------------ #
    # 6. SVD  (S = U Σ Vᵀ)
    # ------------------------------------------------------------------ #
    U, Sigma, VT = np.linalg.svd(S, full_matrices=False)

    # ------------------------------------------------------------------ #
    # 7. Eigenvalues, inertia
    # ------------------------------------------------------------------ #
    eigenvalues = Sigma ** 2
    # Drop the trivial first eigenvalue (≈0 artefact of centring)
    # but keep all for transparency; filter positive ones only
    n_axes = min(n_rows - 1, n_cols - 1, len(eigenvalues))
    n_axes = max(n_axes, 1)

    eigenvalues_used = eigenvalues[:n_axes]
    total_inertia = float(np.sum(eigenvalues_used))
    if total_inertia == 0:
        total_inertia = 1e-10
    explained_inertia = (eigenvalues_used / total_inertia * 100).tolist()

    # ------------------------------------------------------------------ #
    # 8. Principal coordinates
    # Rows:    F = Di^(-1/2) U Σ
    # Columns: G = Dj^(-1/2) Vᵀᵀ Σ
    # ------------------------------------------------------------------ #
    C_row = Di_inv_sqrt @ U[:, :n_axes] @ np.diag(Sigma[:n_axes])   # (n_rows, n_axes)
    C_col = Dj_inv_sqrt @ VT[:n_axes, :].T @ np.diag(Sigma[:n_axes])  # (n_cols, n_axes)

    # ------------------------------------------------------------------ #
    # 9. Contributions & cos² (quality of representation)
    # ------------------------------------------------------------------ #
    # Row contributions to each axis
    row_contrib = (fi_safe[:, None] * C_row ** 2) / eigenvalues_used[None, :]  # (n_rows, n_axes)
    # Col contributions to each axis
    col_contrib = (fj_safe[:, None] * C_col ** 2) / eigenvalues_used[None, :]  # (n_cols, n_axes)

    # Cos² (squared cosine = quality of representation)
    row_dist2 = np.sum(C_row ** 2, axis=1, keepdims=True)
    col_dist2 = np.sum(C_col ** 2, axis=1, keepdims=True)
    row_cos2 = C_row ** 2 / np.where(row_dist2 == 0, 1e-10, row_dist2)
    col_cos2 = C_col ** 2 / np.where(col_dist2 == 0, 1e-10, col_dist2)

    axis_labels = [f"F{i + 1}" for i in range(n_axes)]

    return {
        # Raw data
        "row_labels": row_labels,
        "col_labels": col_labels,
        "X": X.tolist(),
        "n": float(n),
        # Frequencies
        "fij": fij.tolist(),
        "fi": fi.tolist(),
        "fj": fj.tolist(),
        # Residuals & SVD
        "S": S.tolist(),
        "U": U[:, :n_axes].tolist(),
        "VT": VT[:n_axes, :].tolist(),
        "Sigma": Sigma[:n_axes].tolist(),
        # Results
        "eigenvalues": eigenvalues_used.tolist(),
        "total_inertia": total_inertia,
        "explained_inertia": explained_inertia,
        "cumulative_inertia": np.cumsum(explained_inertia).tolist(),
        "n_axes": n_axes,
        "axis_labels": axis_labels,
        # Coordinates
        "C_row": C_row.tolist(),
        "C_col": C_col.tolist(),
        # Quality stats
        "row_contrib": (row_contrib * 100).tolist(),
        "col_contrib": (col_contrib * 100).tolist(),
        "row_cos2": row_cos2.tolist(),
        "col_cos2": col_cos2.tolist(),
    }


def build_afc_tables(result):
    """
    Convert raw numpy output into display-ready pandas HTML tables.
    """
    row_labels = result["row_labels"]
    col_labels = result["col_labels"]
    axis_labels = result["axis_labels"]

    def _html(df, decimals=4):
        return df.round(decimals).to_html(classes="table", border=0)

    # Contingency table
    contingency_html = _html(
        pd.DataFrame(result["X"], index=row_labels, columns=col_labels), 0
    )

    # Relative frequencies
    freq_html = _html(
        pd.DataFrame(result["fij"], index=row_labels, columns=col_labels), 4
    )

    # Masses
    masses_row_html = _html(
        pd.DataFrame({"Masse (ri)": result["fi"]}, index=row_labels), 4
    )
    masses_col_html = _html(
        pd.DataFrame({"Masse (cj)": result["fj"]}, index=col_labels), 4
    )

    # S matrix
    S_html = _html(
        pd.DataFrame(result["S"], index=row_labels, columns=col_labels), 4
    )

    # SVD
    U_html = _html(
        pd.DataFrame(result["U"], index=row_labels,
                     columns=[f"u{i+1}" for i in range(len(result["U"][0]))]), 4
    )
    VT_html = _html(
        pd.DataFrame(result["VT"],
                     index=[f"v{i+1}" for i in range(len(result["VT"]))],
                     columns=col_labels), 4
    )
    sigma_html = _html(
        pd.DataFrame({"σ": result["Sigma"],
                      "Valeur propre (σ²)": result["eigenvalues"],
                      "Inertie expliquée (%)": result["explained_inertia"],
                      "Inertie cumulée (%)": result["cumulative_inertia"]}), 4
    )

    # Coordinates
    coord_row_html = _html(
        pd.DataFrame(result["C_row"], index=row_labels, columns=axis_labels), 4
    )
    coord_col_html = _html(
        pd.DataFrame(result["C_col"], index=col_labels, columns=axis_labels), 4
    )

    # Contributions
    contrib_row_html = _html(
        pd.DataFrame(result["row_contrib"], index=row_labels,
                     columns=[f"CTR {a} (%)" for a in axis_labels]), 2
    )
    contrib_col_html = _html(
        pd.DataFrame(result["col_contrib"], index=col_labels,
                     columns=[f"CTR {a} (%)" for a in axis_labels]), 2
    )

    # Cos²
    cos2_row_html = _html(
        pd.DataFrame(result["row_cos2"], index=row_labels,
                     columns=[f"cos² {a}" for a in axis_labels]), 4
    )
    cos2_col_html = _html(
        pd.DataFrame(result["col_cos2"], index=col_labels,
                     columns=[f"cos² {a}" for a in axis_labels]), 4
    )

    return {
        "contingency_html": contingency_html,
        "freq_html": freq_html,
        "masses_row_html": masses_row_html,
        "masses_col_html": masses_col_html,
        "S_html": S_html,
        "U_html": U_html,
        "VT_html": VT_html,
        "sigma_html": sigma_html,
        "coord_row_html": coord_row_html,
        "coord_col_html": coord_col_html,
        "contrib_row_html": contrib_row_html,
        "contrib_col_html": contrib_col_html,
        "cos2_row_html": cos2_row_html,
        "cos2_col_html": cos2_col_html,
    }
