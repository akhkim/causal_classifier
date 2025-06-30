import numpy as np
from causalml.propensity import ElasticNetPropensityModel

def estimate(data, treatment, outcome, covariates,
             n_fold=5, clip=(1e-3, 1-1e-3), random_state=42):
    # 1. Extract arrays
    Y = data[outcome].to_numpy()
    T = data[treatment].to_numpy()
    X = data[covariates].to_numpy()

    # 2. Fit propensity-score model (elastic-net CV logistic regression)
    pm = ElasticNetPropensityModel(n_fold=n_fold,
                                   random_state=random_state,
                                   clip_bounds=clip)
    p_hat = pm.fit_predict(X, T)

    # 4. ATE
    ate_scores = T * Y / p_hat - (1 - T) * Y / (1 - p_hat)
    ate = ate_scores.mean()
    ate_se = ate_scores.std(ddof=1) / np.sqrt(len(Y))

    # 5. ATT
    att_scores = (T - p_hat) * Y / p_hat
    att = att_scores.mean()
    att_se = att_scores.std(ddof=1) / np.sqrt(len(Y))

    return {
        "ATE": float(ate),
        "ATE_std_error": float(ate_se),
        "ATT": float(att),
        "ATT_std_error": float(att_se),
        "model": pm
    }



# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# def estimate(
#     data,
#     treatment,
#     outcome,
#     covariates,
#     effect
# ):
#     df = data.copy()

#     # Validate inputs
#     if treatment not in df.columns or outcome not in df.columns:
#         raise KeyError("Treatment or outcome column not found in data.")
#     for cov in covariates:
#         if cov not in df.columns:
#             raise KeyError(f"Covariate '{cov}' not found in data.")

#     # Fit propensity score model
#     X_ps = sm.add_constant(df[covariates])
#     ps_model = sm.Logit(df[treatment], X_ps).fit(disp=False)
#     pscore = ps_model.predict(X_ps)
#     # Trim extreme scores
#     pscore = np.clip(pscore, 1e-3, 1-1e-3)
#     df['pscore'] = pscore

#     # Compute weights
#     if effect == 'ate':
#         df['weights'] = df[treatment] / df['pscore'] + (1 - df[treatment]) / (1 - df['pscore'])
#     elif effect == 'att':
#         df['weights'] = np.where(
#             df[treatment] == 1,
#             1.0,
#             df['pscore'] / (1 - df['pscore'])
#         )

#     # Weighted regression of outcome on treatment
#     formula = f"{outcome} ~ {treatment}"
#     outcome_model = smf.wls(formula, data=df, weights=df['weights']).fit()
#     estimate = outcome_model.params[treatment]
#     std_error = outcome_model.bse[treatment]

#     return {
#         'estimate': estimate,
#         'std_error': std_error,
#         'ps_model': ps_model,
#         'outcome_model': outcome_model
#     }
