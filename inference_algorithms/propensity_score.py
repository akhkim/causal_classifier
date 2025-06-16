from econml.dr import IPW

def estimate(
    data,
    treatment,
    outcome,
    covariates
):
    Y = data[outcome].values
    T = data[treatment].values
    X = data[covariates]
    ipw = IPW(model_propensity='auto')
    ipw.fit(Y, T, X=X)

    ate_est = ipw.ate(X)
    ate_se = ipw.ate_std(X)
    att_est = ipw.att(X)
    att_se = ipw.att_std(X)
    
    return {
        'ATE': float(ate_est),
        'ATE_std_error': float(ate_se),
        'ATT': float(att_est),
        'ATT_std_error': float(att_se),
        'model': ipw
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
