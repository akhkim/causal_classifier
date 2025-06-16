from doubleml import DoubleMLPLR, DoubleMLData
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

def estimate(
    data,
    treatment,
    outcome,
    covariates,
    sample_size
):
    df = data.copy()
    ml_Q = RandomForestRegressor(n_estimators=100)
    ml_g = LogisticRegression(solver='lbfgs', max_iter=1000)
    if sample_size < 100:
        k = sample_size
    elif 100 <= sample_size < 500:
        k = 20
    else:
        k = 10

    dml_data = DoubleMLPLR(
        data=DoubleMLData(df, y_col=outcome, d_cols=[treatment], x_cols=covariates),
        ml_g=ml_g,
        ml_Q=ml_Q,
        n_folds=k
    )
    dml_data.fit()
    ate = dml_data.ate
    se = dml_data.ate_se
    return {
        'estimate': ate,
        'std_error': se,
        'model': dml_data
    }


# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.base import clone
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression

# def estimate(
#     data,
#     treatment,
#     outcome,
#     covariates,
#     sample_size
# ):
#     ml_Q = RandomForestRegressor(n_estimators=100)
#     ml_g = LogisticRegression(solver='lbfgs', max_iter=1000)
#     if sample_size < 100:
#         n_splits = sample_size
#     elif 100 <= sample_size < 500:
#         n_splits = 20
#     else:
#         n_splits = 10
    
#     T = data[treatment].values
#     Y = data[outcome].values
#     X = data[covariates].values
#     n = len(Y)

#     # Storage for influence values
#     D = np.zeros(n)

#     # Cross-fitting
#     kf = KFold(n_splits=n_splits, shuffle=True)
#     for train_idx, test_idx in kf.split(X):
#         # Fit Q-models on training fold
#         Q1_mod = clone(ml_Q)
#         Q0_mod = clone(ml_Q)
#         Q1_mod.fit(X[train_idx][T[train_idx]==1], Y[train_idx][T[train_idx]==1])
#         Q0_mod.fit(X[train_idx][T[train_idx]==0], Y[train_idx][T[train_idx]==0])

#         # Fit g-model on training fold
#         g_mod = clone(ml_g).fit(X[train_idx], T[train_idx])

#         # Predict nuisances on test fold
#         Q1_hat = Q1_mod.predict(X[test_idx])
#         Q0_hat = Q0_mod.predict(X[test_idx])
#         if hasattr(g_mod, 'predict_proba'):
#             g_hat = g_mod.predict_proba(X[test_idx])[:,1]
#         else:
#             g_hat = g_mod.predict(X[test_idx])
#         g_hat = np.clip(g_hat, 1e-3, 1-1e-3)

#         # Compute influence function
#         Ai = T[test_idx]
#         Yi = Y[test_idx]
#         Di = (
#             Q1_hat - Q0_hat
#             + Ai * (Yi - Q1_hat) / g_hat
#             - (1 - Ai) * (Yi - Q0_hat) / (1 - g_hat)
#         )
#         D[test_idx] = Di

#     # Aggregate results
#     ate = D.mean()
#     se = D.std(ddof=1) / np.sqrt(n)

#     return {'estimate': ate, 'std_error': se, 'influence': D}  # influence: np.ndarray — influence function values
