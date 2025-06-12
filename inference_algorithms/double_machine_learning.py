import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

def double_machine_learning_aiptw(
    data,
    treatment,
    outcome,
    covariates,
    ml_Q,
    ml_g,
    n_splits
):
    # Set default nuisance learners if not provided
    ml_Q = ml_Q or RandomForestRegressor(n_estimators=100)
    ml_g = ml_g or LogisticRegression(solver='lbfgs', max_iter=1000)

    # Extract arrays
    T = data[treatment].values
    Y = data[outcome].values
    X = data[covariates].values
    n = len(Y)

    # Storage for influence values
    D = np.zeros(n)

    # Cross-fitting
    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        # Fit Q-models on training fold
        Q1_mod = clone(ml_Q)
        Q0_mod = clone(ml_Q)
        Q1_mod.fit(X[train_idx][T[train_idx]==1], Y[train_idx][T[train_idx]==1])
        Q0_mod.fit(X[train_idx][T[train_idx]==0], Y[train_idx][T[train_idx]==0])

        # Fit g-model on training fold
        g_mod = clone(ml_g).fit(X[train_idx], T[train_idx])

        # Predict nuisances on test fold
        Q1_hat = Q1_mod.predict(X[test_idx])
        Q0_hat = Q0_mod.predict(X[test_idx])
        if hasattr(g_mod, 'predict_proba'):
            g_hat = g_mod.predict_proba(X[test_idx])[:,1]
        else:
            g_hat = g_mod.predict(X[test_idx])
        g_hat = np.clip(g_hat, 1e-3, 1-1e-3)

        # Compute influence function
        Ai = T[test_idx]
        Yi = Y[test_idx]
        Di = (
            Q1_hat - Q0_hat
            + Ai * (Yi - Q1_hat) / g_hat
            - (1 - Ai) * (Yi - Q0_hat) / (1 - g_hat)
        )
        D[test_idx] = Di

    # Aggregate results
    ate = D.mean()
    se = D.std(ddof=1) / np.sqrt(n)

    return {'estimate': ate, 'std_error': se, 'influence': D}  # influence: np.ndarray — influence function values




# from doubleml import DoubleMLPLR


# def double_machine_learning(data, treatment, outcome, covariates, ml_Q=None, ml_g=None, n_splits=5):
#     dml_data = DoubleMLPLR(
#         pd.DataFrame(data),
#         y_col=outcome,
#         d_cols=[treatment],
#         x_cols=covariates
#     )
#     dml_data.set_ml_nuisances(ml_g, ml_Q)
#     dml_data.fit()
#     ate = dml_data.ate
#     se = dml_data.ate_se
#     return {
#         'estimate': ate,
#         'std_error': se,
#         'model': dml_data
#     }
