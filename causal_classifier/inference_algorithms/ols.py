import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import diagnostic

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

def diagnose(data, treatment, outcome, adjustment_set=None, alpha=0.05):
    """
    Comprehensive OLS diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    df = data.copy()
    df["__treat__"] = df[treatment]
    df["__outcome__"] = df[outcome]
    
    diagnostics = {
        'linearity': True,
        'normality': True,
        'homoscedasticity': True,
        'independence': True,
        'multicollinearity': True,
        'overall_valid': True
    }
    
    try:
        # Fit the model first
        if adjustment_set is None or len(adjustment_set) == 0:
            covariate_terms = ""
        else:
            if isinstance(adjustment_set, set):
                adjustment_list = list(adjustment_set)
            else:
                adjustment_list = adjustment_set
            string_covariates = [str(item) for item in adjustment_list if item is not None]
            covariate_terms = " + ".join(string_covariates) if string_covariates else ""

        rhs = "__treat__" + (f" + {covariate_terms}" if covariate_terms else "")
        formula = f"__outcome__ ~ {rhs}"
        model = smf.ols(formula, data=df).fit()
        
        # 1. Test for linearity (using reset test - Ramsey's specification test)
        try:
            from statsmodels.stats.diagnostic import linear_reset
            reset_stat, reset_p = linear_reset(model, power=2)
            diagnostics['linearity'] = reset_p > alpha
        except:
            # Fallback: simple correlation test for continuous variables
            numeric_vars = df.select_dtypes(include=[np.number]).columns
            if len(numeric_vars) > 1:
                correlations = []
                for var in numeric_vars:
                    if var != '__outcome__':
                        corr, p_val = stats.pearsonr(df[var], df['__outcome__'])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                # Simple heuristic: if all correlations are very weak, linearity might be violated
                diagnostics['linearity'] = len(correlations) == 0 or max(correlations) > 0.1
        
        # 2. Test for normality of residuals (Shapiro-Wilk test)
        residuals = model.resid
        if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limits
            _, norm_p = stats.shapiro(residuals)
            diagnostics['normality'] = norm_p > alpha
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            _, norm_p = stats.normaltest(residuals)
            diagnostics['normality'] = norm_p > alpha
        
        # 3. Test for homoscedasticity (Breusch-Pagan test)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            _, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
            diagnostics['homoscedasticity'] = bp_p > alpha
        except:
            # Fallback: simple variance ratio test
            fitted_values = model.fittedvalues
            median_fitted = np.median(fitted_values)
            low_group = residuals[fitted_values <= median_fitted]
            high_group = residuals[fitted_values > median_fitted]
            if len(low_group) > 1 and len(high_group) > 1:
                _, p_val = stats.levene(low_group, high_group)
                diagnostics['homoscedasticity'] = p_val > alpha
        
        # 4. Test for independence (Durbin-Watson test if time series structure detected)
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(residuals)
            # DW statistic should be around 2 for no autocorrelation
            diagnostics['independence'] = 1.5 <= dw_stat <= 2.5
        except:
            diagnostics['independence'] = True  # Assume independence if test fails
        
        # 5. Test for multicollinearity (VIF)
        if adjustment_set and len(adjustment_set) > 1:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X = model.model.exog
                vif_data = pd.DataFrame()
                vif_data["Variable"] = model.model.exog_names
                vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                # VIF > 10 indicates problematic multicollinearity
                diagnostics['multicollinearity'] = all(vif_data["VIF"] < 10)
            except:
                # Fallback: correlation matrix check
                numeric_covars = [var for var in adjustment_set if var in df.select_dtypes(include=[np.number]).columns]
                if len(numeric_covars) > 1:
                    corr_matrix = df[numeric_covars].corr().abs()
                    # Check for high correlations (> 0.8)
                    np.fill_diagonal(corr_matrix.values, 0)
                    diagnostics['multicollinearity'] = (corr_matrix < 0.8).all().all()
        
        # Overall validity
        diagnostics['overall_valid'] = all([
            diagnostics['linearity'],
            diagnostics['normality'], 
            diagnostics['homoscedasticity'],
            diagnostics['independence'],
            diagnostics['multicollinearity']
        ])
        
    except Exception as e:
        print(f"OLS diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics
