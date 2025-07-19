from linearmodels.iv import IV2SLS

def estimate(
    data, 
    treatment, 
    outcome, 
    instruments, 
    covariates
):
    exog = list(set(covariates) - set(instruments)) if covariates else []
    # build formula: outcome ~ exog + [treatment ~ instruments]
    inst = ' + '.join(instruments)
    ex = ' + '.join(exog) if exog else '1'
    formula = f"{outcome} ~ {ex} + [{treatment} ~ {inst}]"
    
    print(f"DEBUG - Treatment: {treatment}")
    print(f"DEBUG - Outcome: {outcome}")
    print(f"DEBUG - Instruments: {instruments}")
    print(f"DEBUG - Original covariates: {covariates}")
    print(f"DEBUG - Final exog variables: {exog}")

    iv_res = IV2SLS.from_formula(formula, data).fit(cov_type='robust')
    return {
        'estimate': iv_res.params[treatment],
        'std_error': iv_res.std_errors[treatment],
        'model': iv_res
    }
