import statsmodels.formula.api as smf

def ols(data,
        treatment,
        outcome
):
    df = data.copy()
    df['__treat__'] = df[treatment]
    df['__outcome__'] = df[outcome]

    # Build and fit the OLS model
    formula = '__outcome__ ~ __treat__'
    model = smf.ols(formula, data=df).fit()

    # Extract estimate and standard error
    estimate = model.params['__treat__']
    std_err = model.bse['__treat__']

    return {'estimate': estimate, 'std_error': std_err, 'model': model}