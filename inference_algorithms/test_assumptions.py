"""
causal_assumption_tests.py

Extended collection of universal Python functions for testing the *essential* 
identifying assumptions of common causal-inference methods on any pandas DataFrame.
Each function returns True if the algorithm is viable, False if it should be rejected.

Dependencies: pandas, numpy, statsmodels, sklearn, linearmodels, econml
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from sklearn.linear_model import LogisticRegression
from linearmodels.iv import IV2SLS
from econml.dml import LinearDML
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def test_linearity(df, outcome, predictors, *, power=2, use_f=True, alpha=0.05):
    """
    Ramsey RESET test for linearity.
    Returns True if linearity assumption holds (p > alpha), False otherwise.
    """
    try:
        X = sm.add_constant(df[predictors])
        y = df[outcome]
        model = sm.OLS(y, X).fit()
        reset_res = linear_reset(model, power=power, use_f=use_f)
        return reset_res.pvalue > alpha
    except:
        return False

def test_exogeneity(df, outcome, exog, endog, instruments, alpha=0.05):
    """
    Durbin–Wu–Hausman test for endogeneity.
    Returns True if exogeneity holds (p > alpha), False otherwise.
    """
    try:
        if not instruments:
            return False
        
        X1 = sm.add_constant(df[exog + instruments])
        fs = sm.OLS(df[endog], X1).fit()
        df_temp = df.copy()
        df_temp['_resid'] = fs.resid
        X2 = sm.add_constant(df_temp[exog + [endog, '_resid']])
        aug = sm.OLS(df_temp[outcome], X2).fit()
        pval = aug.pvalues['_resid']
        return pval > alpha
    except:
        return False


def test_positivity(df, treatment, covariates, eps=0.05, min_coverage=0.90):
    """
    Empirical overlap test via propensity-score common support.
    Returns True if sufficient overlap exists (coverage > min_coverage), False otherwise.
    """
    try:
        X = df[covariates]
        T = df[treatment]
        lr = LogisticRegression(max_iter=1000).fit(X, T)
        ps = lr.predict_proba(X)[:, 1]
        mask = (ps < eps) | (ps > 1 - eps)
        coverage = 1 - mask.mean()
        return coverage > min_coverage
    except:
        return False


def test_parallel_trends(df, outcome, group_variable, treatment, alpha=0.05):
    """
    Event-study style placebo pre-trends test for DiD.
    Returns True if parallel trends hold (all leads p > alpha), False otherwise.
    """
    try:
        if group_variable == "None" or group_variable is None:
            return False
        
        d = df.copy()
        for lead in range(1, 5):
            d[f'treatment_lead_{lead}'] = (
                d.groupby(group_variable)[treatment]
                .shift(-lead)
                .fillna(0)
                .astype(int)
            )
            d[f'D_p_{lead}'] = d[f'treatment_lead_{lead}'] * d[group_variable]
            X = sm.add_constant(d[[group_variable, f'treatment_lead_{lead}', f'D_p_{lead}']])
            y = d[outcome]
            res = sm.OLS(y, X).fit()
            
            # If any lead shows significant pre-trend, reject
            if res.pvalues[f'D_p_{lead}'] <= alpha:
                return False
        
        return True
    except:
        return False


def test_no_anticipation(df, outcome, group_variable, treatment, alpha=0.05):
    """
    Event-study style no-anticipation test for DiD.
    Returns True if no anticipation detected (all leads p > alpha), False otherwise.
    """
    try:
        if group_variable == "None" or group_variable is None:
            return False
        
        d = df.copy()
        for lead in range(1, 5):
            d[f'treatment_lead_{lead}'] = (
                d.groupby(group_variable)[treatment]
                .shift(-lead)
                .fillna(0)
                .astype(int)
            )
            d[f'D_a_{lead}'] = d[f'treatment_lead_{lead}'] * d[group_variable]
            X = sm.add_constant(d[[group_variable, f'treatment_lead_{lead}', f'D_a_{lead}']])
            y = d[outcome]
            res = sm.OLS(y, X).fit()
            
            # If any lead shows significant anticipation, reject
            if res.pvalues[f'D_a_{lead}'] <= alpha:
                return False
        
        return True
    except:
        return False


def test_rdd_continuity(df, outcome, running_variable, cutoff_value, 
                       bandwidth=None, poly=1, alpha=0.05):
    """
    Continuity test at RDD cutoff.
    Returns True if continuity holds (p > alpha), False otherwise.
    """
    try:
        if running_variable == "None" or running_variable is None:
            return False
        
        df_temp = df.copy()
        if bandwidth is not None:
            df_temp = df_temp[np.abs(df_temp[running_variable] - cutoff_value) <= bandwidth]
        
        df_temp['T'] = (df_temp[running_variable] >= cutoff_value).astype(int)
        df_temp['dist'] = df_temp[running_variable] - cutoff_value
        X = pd.DataFrame({'const': 1, 'T': df_temp['T']})
        
        for p in range(1, poly + 1):
            X[f'dist_{p}'] = df_temp['dist']**p
            X[f'inter_{p}'] = df_temp['T'] * df_temp['dist']**p
        
        y = df_temp[outcome]
        m = sm.OLS(y, X).fit()
        return m.pvalues['T'] > alpha
    except:
        return False


def test_iv_strength(df, treatment, instruments, covariates):   # Cautionary test. Doesn't invalidate IVs even if weak.
    """
    First-stage F-statistic for instrument relevance.
    Returns True if instruments are strong (F > min_f), False otherwise.
    """
    if not instruments:
        return False
    
    X = sm.add_constant(df[covariates + instruments])
    fs = sm.OLS(df[treatment], X).fit()
    k = len(fs.params)
    q = len(instruments)
    
    instr_positions = [fs.params.index.get_loc(instr) for instr in instruments]
    R = np.zeros((q, k))
    for i, pos in enumerate(instr_positions):
        R[i, pos] = 1.0
    
    zeros = np.zeros(q)
    ftest = fs.f_test((R, zeros))
    return float(ftest.fvalue)


def test_iv_overid(df, outcome, treatment, instruments, exog_vars, alpha=0.05):
    """
    Hansen J-test for overidentification.
    Returns True if instruments are valid (p > alpha), False otherwise.
    """
    try:
        if not instruments or len(instruments) <= 1:
            return True  # Cannot test with <= 1 instrument, assume valid
        
        iv = IV2SLS(df[outcome], sm.add_constant(df[exog_vars]), 
                   df[treatment], df[instruments]).fit()
        return iv.j_stat.pval > alpha
    except:
        return False


def test_frontdoor_complete_mediation(df, outcome, treatment, mediators, 
                                     covariates, alpha=0.05):
    """
    Test for complete mediation in front-door design.
    Returns True if direct effect vanishes (p > alpha), False otherwise.
    """
    try:
        if not mediators:
            return False
        
        covariates = [] if covariates is None else covariates
        if isinstance(mediators, str):
            mediators = [mediators]
        
        # Full model with mediators
        cols_full = [treatment] + mediators + covariates
        X_full = sm.add_constant(df[cols_full])
        m1 = sm.OLS(df[outcome], X_full).fit()
        p_direct = m1.pvalues[treatment]
        
        return p_direct > alpha
    except:
        return False


def test_frontdoor_no_backdoor(df, mediators, covariates, alpha=0.05):
    """
    Test for no back-door confounding into mediators.
    Returns True if no back-door paths detected, False otherwise.
    """
    try:
        if not mediators:
            return False
        
        for M in mediators:
            X = sm.add_constant(df[covariates])
            y = df[M]
            model = sm.OLS(y, X).fit()
            
            # Check if any covariate significantly predicts mediator
            for cov in covariates:
                if model.pvalues[cov] <= alpha:
                    return False  # Back-door path detected
        
        return True
    except:
        return False


def test_dml_crossfit_stability(df, outcome, treatment, covariates, 
                               n_splits=5, random_state=0, max_std=0.1):
    """
    Check stability of DML estimate across cross-fitting splits.
    Returns True if estimates are stable (std < max_std), False otherwise.
    """
    try:
        estimates = []
        for i in range(n_splits):
            est = LinearDML(random_state=random_state + i)
            est.fit(Y=df[outcome], T=df[treatment], X=df[covariates], W=None)
            estimates.append(est.effect(df[covariates]).mean())
        
        std_effect = np.std(estimates)
        return std_effect < max_std
    except:
        return False


def test_gcomp_rmse(df, outcome, covariates, max_rmse=None):
    """
    Test G-computation model specification via RMSE.
    Returns True if model fit is adequate, False otherwise.
    """
    try:
        X = df[covariates]
        y = df[outcome]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        preds = model.predict(sm.add_constant(X))
        rmse = np.sqrt(mean_squared_error(y, preds))
        
        # If no threshold provided, use outcome standard deviation as benchmark
        if max_rmse is None:
            max_rmse = y.std() * 0.5  # RMSE should be < 50% of outcome std
        
        return rmse < max_rmse
    except:
        return False


def run_all_tests(nx_graph, df, treatment, outcome, covariates, group_variable, 
                 instruments, cutoff_value, running_variable, mediators):
    """
    Run all diagnostic tests and return boolean results.
    """
    results = {}
    
    try:
        endo_vars = [v for v in covariates if nx_graph.in_degree(v) > 0]
        exog_vars = [v for v in covariates if nx_graph.in_degree(v) == 0]
    except:
        exog_vars = covariates
        endo_vars = []
    
    results['linearity'] = test_linearity(df, outcome, covariates)
    results['exogeneity'] = test_exogeneity(df, outcome, covariates, treatment, instruments)
    results['positivity'] = test_positivity(df, treatment, covariates)
    results['parallel_trends'] = test_parallel_trends(df, outcome, group_variable, treatment)
    results['no_anticipation'] = test_no_anticipation(df, outcome, group_variable, treatment)
    results['rdd_continuity'] = test_rdd_continuity(df, outcome, running_variable, cutoff_value)
    results['iv_strength'] = test_iv_strength(df, treatment, instruments, exog_vars)
    
    if len(instruments) > len(endo_vars):
        results['iv_overid'] = test_iv_overid(df, outcome, treatment, instruments, exog_vars)
    
    results['frontdoor_mediation'] = test_frontdoor_complete_mediation(df, outcome, treatment, mediators, covariates)
    results['frontdoor_no_backdoor'] = test_frontdoor_no_backdoor(df, mediators, covariates)
    results['dml_stability'] = test_dml_crossfit_stability(df, outcome, treatment, covariates)
    results['gcomp_rmse'] = test_gcomp_rmse(df, outcome, covariates)
    
    print(results)
    return results
