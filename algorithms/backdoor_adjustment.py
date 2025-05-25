import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def backdoor_adjustment(
    data,
    treatment,
    outcome,
    covariates,
    effect = 'ate'
):
    df = data.copy()

    # Validate inputs
    if treatment not in df.columns or outcome not in df.columns:
        raise KeyError("Treatment or outcome column not found in data.")
    for c in covariates:
        if c not in df.columns:
            raise KeyError(f"Covariate '{c}' not found in data.")

    # Build formula for regression
    base_formula = f"{outcome} ~ {treatment}"
    if covariates:
        base_formula += " + " + " + ".join(covariates)

    # ATE via OLS
    if effect == 'ate':
        model = smf.ols(base_formula, data=df).fit()
        est = model.params[treatment]
        se  = model.bse[treatment]

    # ATT via propensity-score weighting
    elif effect == 'att':
        # Fit propensity score model
        X_ps = sm.add_constant(df[covariates])
        ps_model = sm.Logit(df[treatment], X_ps).fit(disp=False)
        pscore = ps_model.predict(X_ps)
        # Compute weights: treated=1, control=p/(1-p)
        weights = np.where(df[treatment] == 1, 1.0, pscore / (1 - pscore))
        # Weighted regression
        model = smf.wls(base_formula, data=df, weights=weights).fit()
        est = model.params[treatment]
        se  = model.bse[treatment]

    return {
        'estimate': est,
        'std_error': se,
        'model': model
    }




# from dowhy import CausalModel

# def backdoor_adjustment(data, treatment, outcome, covariates, effect='ate'):
#     model = CausalModel(
#         data=data,
#         treatment=treatment,
#         outcome=outcome,
#         common_causes=covariates
#     )
#     identified = model.identify_effect()
#     method = 'backdoor.linear_regression' if effect=='ate' else 'backdoor.propensity_score_weighting'
#     est = model.estimate_effect(
#         identified,
#         method_name=method
#     )
#     return {
#         'estimate': est.value,
#         'std_error': est.get_std_error(),
#         'model': est
#     }