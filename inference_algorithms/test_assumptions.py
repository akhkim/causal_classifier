"""
causal_assumption_tests.py

Extended collection of universal Python functions for testing the *essential* 
identifying assumptions of common causal-inference methods on any pandas DataFrame:
  - Linearity (Ramsey RESET)
  - Exogeneity (Durbin–Wu–Hausman)
  - Positivity / Overlap (propensity‐score common support)
  - Difference-in-Differences (parallel trends + no anticipation)
  - Regression Discontinuity (continuity at cutoff)
  - Instrumental Variables (relevance + overidentification)
  - Front-door Adjustment (complete mediation + no back-door into M or Y)
  - Double Machine Learning (Neyman orthogonality + cross-fitting stability)
  - G-computation (correct outcome model specification via RMSE check)
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


def test_linearity(df, outcome, predictors, *, power=2, use_f=True):
    """
    Ramsey RESET test for linearity.
    Returns F-statistic and p-value under H0: correct linear specification.
    """
    X = sm.add_constant(df[predictors])
    y = df[outcome]
    model = sm.OLS(y, X).fit()
    reset_res = linear_reset(model, power=power, use_f=use_f)
    return {'f_stat': reset_res.fvalue, 'p_value': reset_res.pvalue}


def test_exogeneity(df, outcome,
                    exog, endog, instruments):
    """
    Durbin–Wu–Hausman test for endogeneity of `endog`.
    Essential for OLS.
    """
    if instruments == []:
        return "No valid instruments found"
    
    X1 = sm.add_constant(df[exog + instruments])
    fs = sm.OLS(df[endog], X1).fit()
    df['_resid'] = fs.resid
    X2 = sm.add_constant(df[exog + [endog, '_resid']])
    aug = sm.OLS(df[outcome], X2).fit()
    stat = aug.tvalues['_resid']
    pval = aug.pvalues['_resid']
    df.drop(columns=['_resid'], inplace=True)
    return {'t_stat': stat, 'p_value': pval}


def test_positivity(df, treatment, covariates, eps: float = 0.05):
    """
    Empirical overlap test via propensity‐score common support.
    Essential for propensity-score methods.
    """
    X = df[covariates]
    T = df[treatment]
    lr = LogisticRegression(max_iter=1000).fit(X, T)
    ps = lr.predict_proba(X)[:, 1]
    mask = (ps < eps) | (ps > 1 - eps)
    coverage = 1 - mask.mean()
    return {'coverage': coverage, 'violations': df.index[mask].tolist()}


def test_parallel_trends(df, outcome, group_variable, treatment):
    """
    Event-study style placebo pre-trends test for DiD.
    Assumes `treatment`==1 only post-period for treated units.
    Loops from lead=1 to max_lead and returns dict of results.
    """
    if group_variable == "None":
        return "No group variable found"
    
    results = {}
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
        results[f'lead_{lead}'] = {
            'coef':    res.params[f'D_p_{lead}'],
            'std_err': res.bse[f'D_p_{lead}'],
            'p_value': res.pvalues[f'D_p_{lead}']
        }
    return results


def test_no_anticipation(df, outcome, group_variable, treatment):
    """
    Event-study style no-anticipation test for DiD.
    Assumes `treatment`==1 only post-period for treated units.
    Loops from lead=1 to max_lead and returns dict of results.
    """
    if group_variable == "None":
        return "No group variable found"
    
    results = {}
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
        results[f'lead_{lead}'] = {
            'coef':    res.params[f'D_a_{lead}'],
            'std_err': res.bse[f'D_a_{lead}'],
            'p_value': res.pvalues[f'D_a_{lead}']
        }
    return results


def test_rdd_continuity(df, outcome, running_variable,
                        cutoff_value, bandwidth = None,
                        poly = 1):
    """
    Continuity of conditional mean at cutoff via local regression jump test.
    Essential for RDD.
    """
    if running_variable == "None":
        return "No running variable found"
    
    df = df.copy()
    if bandwidth is not None:
        df = df[np.abs(df[running_variable] - cutoff_value) <= bandwidth]
    df['T'] = (df[running_variable] >= cutoff_value).astype(int)
    df['dist'] = df[running_variable] - cutoff_value
    X = pd.DataFrame({'const': 1, 'T': df['T']})
    for p in range(1, poly + 1):
        X[f'dist_{p}'] = df['dist']**p
        X[f'inter_{p}'] = df['T'] * df['dist']**p
    y = df[outcome]
    m = sm.OLS(y, X).fit()
    return {'coef': m.params['T'], 'std_err': m.bse['T'], 'p_value': m.pvalues['T']}


# def test_iv_strength(df, treatment, instruments, covariates):
#     """
#     First-stage F-statistic for instrument relevance.
#     Essential for IV.
#     """
#     if instruments == []:
#         return "No valid instruments found"

#     X = sm.add_constant(df[covariates + instruments])
#     fs = sm.OLS(df[treatment], X).fit()
#     ftest = fs.f_test(np.eye(len(fs.params))[ [fs.params.index.get_loc(i) for i in instruments] ], np.zeros(len(instruments)))
#     return {'F_stat': float(ftest.fvalue), 'p_value': float(ftest.pvalue)}


def test_iv_overid(df, outcome, treatment, instruments, covariates):
    """
    Hansen J-test for overidentification.
    Essential for IV when >1 instrument.
    """
    if instruments == []:
        return "No valid instruments found"
    
    iv = IV2SLS(df[outcome], sm.add_constant(df[covariates]), df[treatment], df[instruments]).fit()
    return {'J_stat': iv.j_stat.stat, 'p_value': iv.j_stat.pval}


def test_frontdoor_complete_mediation(df, outcome, treatment, mediators, covariates):
    """
    Test that the *direct* effect of treatment on outcome vanishes
    once *all* mediators enter the model. Essential for front-door.
    
    Returns:
      - direct_coef: coefficient on treatment in the full model
      - p_value: its two-sided p-value
      - base_coef, base_p: treatment coef and p-value when mediators omitted
    """
    if mediators == []:
        return "No valid mediators found"
    
    covariates = [] if covariates is None else covariates
    if isinstance(mediators, str):
        mediators = [mediators]
    
    cols_base = [treatment] + covariates
    X_base = sm.add_constant(df[cols_base])
    m0 = sm.OLS(df[outcome], X_base).fit()
    base_coef = m0.params[treatment]
    base_p   = m0.pvalues[treatment]
    
    cols_full = [treatment] + mediators + covariates
    X_full = sm.add_constant(df[cols_full])
    m1 = sm.OLS(df[outcome], X_full).fit()
    direct_coef = m1.params[treatment]
    p_direct    = m1.pvalues[treatment]
    
    return {
        'base_coef':   base_coef,
        'base_p':      base_p,
        'direct_coef': direct_coef,
        'p_value':     p_direct
    }


def test_frontdoor_no_backdoor(df, mediators, covariates):
    """
    For each mediator M in `mediators`, regress M on covariates.
    Any significant covariate coefficient indicates a back-door path
    into M (violating the no-backdoor assumption).
    Essential for front-door.
    """
    if mediators == []:
        return "No valid mediators found"
    
    results = {}
    for M in mediators:
        X = sm.add_constant(df[covariates])
        y = df[M]
        model = sm.OLS(y, X).fit()
        # Save all covariate p-values for this mediator
        pvals = {cov: model.pvalues[cov] for cov in covariates}
        results[M] = {
            'mediator_R2': model.rsquared,
            'covariate_pvalues': pvals
        }
    return results


def test_dml_crossfit_stability(df, outcome, treatment,
                                covariates, n_splits: int = 5,
                                random_state: int = 0):
    """
    Check stability of DML estimate across cross-fitting splits.
    Essential for Double ML.
    """
    estimates = []
    for i in range(n_splits):
        est = LinearDML(random_state=random_state + i)
        est.fit(df[covariates], df[treatment], df[outcome])
        estimates.append(est.effect(df[covariates]))
    return {'mean_effect': np.mean(estimates), 'std_effect': np.std(estimates)}


def test_gcomp_rmse(df, outcome, covariates):
    """
    Fit outcome model and report RMSE as proxy for correct E[Y|X] specification.
    Essential proxy for G-computation model correctness.
    """
    X = df[covariates]
    y = df[outcome]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    preds = model.predict(sm.add_constant(X))
    rmse = np.sqrt(mean_squared_error(y, preds))
    return {'rmse': rmse}


def run_all_tests(df, treatment, outcome, covariates, group_variable, instruments, cutoff_value, running_variable, mediators):
    print(test_linearity(df, outcome, covariates))
    print(test_exogeneity(df, outcome, covariates, treatment, instruments))
    print(test_positivity(df, treatment, covariates))
    print(test_parallel_trends(df, outcome, group_variable, treatment))
    print(test_no_anticipation(df, outcome, group_variable, treatment))
    print(test_rdd_continuity(df, outcome, running_variable, cutoff_value))
    # print(test_iv_strength(df, treatment, instruments, covariates))
    print(test_iv_overid(df, outcome, treatment, instruments, covariates))
    print(test_frontdoor_complete_mediation(df, outcome, treatment, mediators, covariates))
    print(test_frontdoor_no_backdoor(df, mediators, covariates))
    print(test_dml_crossfit_stability(df, outcome, treatment, covariates))
    print(test_gcomp_rmse(df, outcome, covariates))
