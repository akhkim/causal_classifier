import statsmodels.formula.api as smf

def estimate(data, treatment, outcome, adjustment_set):
    df = data.copy()
    df["__treat__"] = df[treatment]
    df["__outcome__"] = df[outcome]

    # Handle None or empty adjustment_set
    if adjustment_set is None or len(adjustment_set) == 0:
        covariate_terms = ""
    else:
        # Convert set to list if necessary and ensure all elements are strings
        if isinstance(adjustment_set, set):
            adjustment_list = list(adjustment_set)
        else:
            adjustment_list = adjustment_set
        
        # Convert all elements to strings and filter out any None values
        string_covariates = [str(item) for item in adjustment_list if item is not None]
        
        # Create the covariate terms string
        covariate_terms = " + ".join(string_covariates) if string_covariates else ""

    # Build the formula
    rhs = "__treat__" + (f" + {covariate_terms}" if covariate_terms else "")
    formula = f"__outcome__ ~ {rhs}"

    model = smf.ols(formula, data=df).fit()

    return {
        "estimate": model.params["__treat__"],
        "std_error": model.bse["__treat__"],
        "model": model
    }
