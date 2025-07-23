from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def estimate(
    data,
    treatment,
    outcome,
    time_variable,
    group_variable,
    covariates=None
):
    df = data.copy()
    panel = df.set_index([group_variable, time_variable])
    exog = [treatment]
    if covariates:
        exog += covariates
    exog = sm.add_constant(panel[exog])
    endog = panel[outcome]
    mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True).fit()
    return {
        'estimate': mod.params[treatment],
        'std_error': mod.std_errors[treatment],
        'model': mod
    }
