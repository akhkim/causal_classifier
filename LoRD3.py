import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

# Result container
LoRD3Result = namedtuple("LoRD3Result", ["variable", "cutoff", "score"])


def standardize(df, cols):
    """Standardize numeric columns to zero mean, unit variance."""
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def compute_normal_llr(T, fX, neigh_idx):
    """
    Compute the Anderson–Darling–style LLR for a candidate neighborhood.
    T: array of treatment values in neighborhood
    fX: array of fitted values f(x) in neighborhood
    neigh_idx: indices of neighborhood points
    Returns: LLR statistic
    """
    # Residuals
    r = T - fX
    # Estimate local noise from residual variance
    sigma2 = r.var(ddof=1) if r.size > 1 else 1.0
    # Under H0: single mean β0
    beta0 = r.mean()
    # Under H1: two means on either side of center
    zc = fX[neigh_idx == neigh_idx][0]  # dummy for notation
    left = r[fX < beta0]
    right = r[fX >= beta0]
    # If any side empty, return zero
    if left.size < 1 or right.size < 1:
        return 0.0
    beta_l, beta_r = left.mean(), right.mean()
    # LLR: difference in Normal log‐likelihoods
    ll0 = -0.5 * np.sum((r - beta0) ** 2) / sigma2
    ll1 = -0.5 * (np.sum((left - beta_l) ** 2) + np.sum((right - beta_r) ** 2)) / sigma2
    return ll1 - ll0


def find_best_neighborhood(z, T, X, poly_degree=1, k=50):
    """
    Scan all local neighborhoods in z to find the one with maximal LLR.
    z: 1D array of candidate forcing variable
    T: 1D array of treatment values
    X: DataFrame of covariates used to fit f(x)
    poly_degree: degree of polynomial for f(x)
    k: neighborhood size
    Returns: (best_score, best_cutoff)
    """
    n = len(z)
    # Fit smooth model f(x) using polynomial regression
    # Design matrix: polynomial features of all covariates in X
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        PolynomialFeatures(poly_degree, include_bias=True),
        LinearRegression()
    )
    model.fit(X, T)
    fX_all = model.predict(X)

    best_score, best_cutoff = -np.inf, None
    # Pre-sort by z for efficient neighborhood selection
    sort_idx = np.argsort(z)
    z_sorted, T_sorted, fX_sorted = z[sort_idx], T[sort_idx], fX_all[sort_idx]

    for i in range(n):
        # Identify k-nearest neighbors in z-space
        center = z_sorted[i]
        distances = np.abs(z_sorted - center)
        neigh_i = np.argsort(distances)[:k]
        T_neigh = T_sorted[neigh_i]
        fX_neigh = fX_sorted[neigh_i]
        # Split neighborhood at cutoff = center
        score = compute_normal_llr(T_neigh, fX_neigh, z_sorted[neigh_i])
        if score > best_score:
            best_score, best_cutoff = score, center

    return best_score, best_cutoff


def detect(df, treatment_col, covariate_cols, poly_degree=1, k=50):
    """
    Automatically detect the best running variable and cutoff.
    df: pandas DataFrame with mixed dtypes
    treatment_col: name of binary or continuous treatment column
    covariate_cols: list of columns to use for fitting f(x)
    poly_degree: degree of polynomial model
    k: neighborhood size
    Returns: LoRD3Result(variable, cutoff, score)
    """
    # Identify numeric columns as candidate running variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(treatment_col)
    df = df.copy().dropna(subset=numeric_cols.append(pd.Index([treatment_col])))
    df = standardize(df, covariate_cols + list(numeric_cols))

    best = LoRD3Result(variable=None, cutoff=None, score=-np.inf)
    X = df[covariate_cols].values
    T = df[treatment_col].values

    for var in numeric_cols:
        z = df[var].values
        score, cutoff = find_best_neighborhood(z, T, X, poly_degree, k)
        if score > best.score:
            best = LoRD3Result(variable=var, cutoff=cutoff, score=score)

    return best
