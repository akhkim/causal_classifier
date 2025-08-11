from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats

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

def diagnose(data, group_variable, time_variable, treatment, outcome, alpha=0.05):
    """
    Comprehensive DiD diagnostic tests.
    
    Returns:
    --------
    bool : True if all key assumptions are satisfied
    """
    df = data.copy()
    
    try:
        # 1. Check for balanced panel structure
        panel_counts = df.groupby([group_variable, time_variable]).size()
        if not all(panel_counts == 1):
            print("Warning: Unbalanced panel detected")
            return False
        
        # 2. Check for sufficient variation in treatment
        treatment_var = df.groupby([group_variable, time_variable])[treatment].nunique()
        if not (treatment_var > 1).any():
            print("Warning: No treatment variation across groups/time")
            return False
        
        # 3. Test for parallel trends (simplified pre-treatment test)
        pre_treatment_periods = df[df[treatment] == 0]
        if len(pre_treatment_periods) < 2:
            print("Warning: Insufficient pre-treatment periods for parallel trends test")
            return False
        
        # Basic parallel trends test using interaction terms
        pre_df = pre_treatment_periods.copy()
        if len(pre_df[time_variable].unique()) >= 2:
            # Simple trend test: group-specific linear trends should be similar
            trend_results = []
            for group in pre_df[group_variable].unique():
                group_data = pre_df[pre_df[group_variable] == group]
                if len(group_data) >= 3:  # Need at least 3 points for trend
                    # Simple linear trend coefficient
                    time_numeric = pd.to_numeric(group_data[time_variable])
                    slope, _, _, p_val, _ = stats.linregress(time_numeric, group_data[outcome])
                    trend_results.append(slope)
            
            if len(trend_results) >= 2:
                # Test if trends are significantly different using F-test proxy
                trend_var = np.var(trend_results)
                if trend_var > np.mean([abs(t) for t in trend_results]) * 0.5:  # Heuristic threshold
                    print("Warning: Parallel trends assumption may be violated")
                    return False
        
        # 4. Check for anticipation effects (treatment should not affect pre-treatment outcomes)
        post_treatment = df[df[treatment] == 1]
        if len(post_treatment) > 0:
            # Check if treatment groups had different outcomes in pre-period
            treated_groups = post_treatment[group_variable].unique()
            pre_treated = pre_treatment_periods[pre_treatment_periods[group_variable].isin(treated_groups)]
            pre_control = pre_treatment_periods[~pre_treatment_periods[group_variable].isin(treated_groups)]
            
            if len(pre_treated) > 0 and len(pre_control) > 0:
                # Test for significant differences in pre-treatment outcomes
                _, p_val = stats.ttest_ind(pre_treated[outcome], pre_control[outcome])
                if p_val < alpha:
                    print("Warning: Pre-treatment differences detected (anticipation effects)")
                    return False
        
        return True
        
    except Exception as e:
        print(f"DiD diagnostic failed: {e}")
        return False
