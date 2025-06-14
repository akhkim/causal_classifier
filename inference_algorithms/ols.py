import statsmodels.formula.api as smf

def estimate(data,
        treatment,
        outcome,
        adjustment_set
):
    df = data.copy()
    df["__treat__"] = df[treatment]
    df["__outcome__"] = df[outcome]

    covariate_terms = " + ".join(adjustment_set)
    rhs = "__treat__" + (f" + {covariate_terms}" if covariate_terms else "")
    formula = f"__outcome__ ~ {rhs}"

    model = smf.ols(formula, data=df).fit()

    return {
        "estimate":  model.params["__treat__"],
        "std_error": model.bse["__treat__"],
        "model":     model
    }