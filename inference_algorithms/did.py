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

def diagnose(data, unit_id, time, treatment, outcome):
    """
    Minimal pre-trend test for a two-way-fixed-effects DiD.
    """
    # a. Need both treated & control and ≥2 pre-treatment periods
    treated_units = data.loc[data[treatment] == 1, unit_id].unique()
    control_units = data.loc[data[treatment] == 0, unit_id].unique()
    if len(treated_units) == 0 or len(control_units) == 0:
        return False
    
    # Construct pre-period subset
    first_treat_time = data.loc[data[treatment] == 1, time].min()
    pre_data = data.loc[data[time] < first_treat_time]
    if pre_data[time].nunique() < 1:
        return False
    
    # b. Simple “parallel trends” placebo regression
    pre_data["trend"] = pre_data[time] - pre_data[time].min()
    pre_data["treated"] = pre_data[unit_id].isin(treated_units).astype(int)
    pre_data["trend_treated"] = pre_data["trend"] * pre_data["treated"]
    
    mod = sm.OLS(pre_data[outcome], sm.add_constant(pre_data[["trend",
                                                         "treated",
                                                         "trend_treated"]]))
    beta = mod.fit().params["trend_treated"]
    pval = mod.fit().pvalues["trend_treated"]
    return pval > 0.10




# import statsmodels.formula.api as smf

# def estimate(
#     data,
#     treatment,
#     outcome,
#     time_variable,
#     covariates=None
# ):
#     """
#     - Assumes treatment is a between-unit grouping (same for both periods).
#     - Infers pre vs. post-period by sorting the two unique values of `time_variable`.
#     """
#     # Check exactly two time periods
#     unique_times = data[time_variable].unique()
#     if len(unique_times) != 2:
#         raise ValueError(f"Expected exactly 2 unique values in '{time_variable}', got {len(unique_times)}.")
#     sorted_times = sorted(unique_times)
#     post_time = sorted_times[1]

#     # Create indicators in a temp copy of data
#     data = data.copy()
#     data['__post__'] = (data[time_variable] == post_time).astype(int)
#     data['__treat__'] = treatment
#     data['__y__'] = outcome
#     data['__did__'] = data['__treat__'] * data['__post__']

#     # Build regression formula
#     formula = "__y__ ~ __treat__ + __post__ + __did__"
#     if covariates:
#         formula += " + " + " + ".join(covariates)

#     # Fit OLS model
#     model = smf.ols(formula, data=data).fit()
#     estimate = model.params['__did__']
#     std_err = model.bse['__did__']

#     return {'estimate': estimate, 'std_error': std_err, 'model': model}

