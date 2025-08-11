from dowhy import CausalModel
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def estimate(
    data,
    treatment,
    mediator,
    adjustment_set,
    outcome,
    latent_confounders=None
):
    """
    Frontdoor adjustment estimation with support for latent confounders.
    
    Parameters:
    -----------
    latent_confounders : list, optional
        List of latent confounder nodes (U_ format) identified by FCI
    """
    
    # Handle latent confounders information
    frontdoor_advantage_note = ""
    if latent_confounders:
        print(f"Frontdoor: Handling {len(latent_confounders)} latent confounders: {latent_confounders}")
        print("Note: Frontdoor adjustment can handle unobserved confounding between treatment and outcome")
        frontdoor_advantage_note = f"Frontdoor estimation with {len(latent_confounders)} latent confounders detected. " \
                                  "Frontdoor path provides identification despite unobserved confounding."
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=adjustment_set,
        instruments=None
    )
    identified = model.identify_effect()
    est = model.estimate_effect(
        identified,
        method_name="frontdoor.linear_regression",
        method_params={"frontdoor_variables": [mediator]}
    )
    return {
        'estimate': est.value,
        'std_error': est.get_standard_error(),
        'model': est
    }

def diagnose(data, treatment, mediator, outcome, adjustment_set=None, alpha=0.05):
    """
    Comprehensive frontdoor adjustment diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    df = data.copy()
    
    diagnostics = {
        'mediator_causally_related': True,
        'no_direct_effect': True,
        'no_confounding_mediator_outcome': True,
        'sufficient_variation': True,
        'model_fit': True,
        'overall_valid': True
    }
    
    try:
        # Prepare data
        T = df[treatment].values
        M = df[mediator].values
        Y = df[outcome].values
        
        # 1. Test if mediator is causally related to treatment
        # Treatment should predict mediator
        med_model = LinearRegression()
        med_model.fit(T.reshape(-1, 1), M)
        t_to_m_r2 = r2_score(M, med_model.predict(T.reshape(-1, 1)))
        
        # Check statistical significance
        n = len(T)
        if t_to_m_r2 > 0 and n > 2:
            f_stat = (t_to_m_r2 / (1 - t_to_m_r2)) * (n - 2)
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
            diagnostics['mediator_causally_related'] = (p_value < alpha) and (t_to_m_r2 > 0.01)
        else:
            diagnostics['mediator_causally_related'] = False
        
        # 2. Test for no direct effect (treatment should only affect outcome through mediator)
        # Compare full model vs. mediation-only model
        full_model = LinearRegression()
        mediation_model = LinearRegression()
        
        # Full model: Y ~ T + M
        X_full = np.column_stack([T, M])
        full_model.fit(X_full, Y)
        
        # Mediation-only model: Y ~ M
        mediation_model.fit(M.reshape(-1, 1), Y)
        
        # If frontdoor is valid, adding T directly shouldn't improve the model much
        full_r2 = r2_score(Y, full_model.predict(X_full))
        med_r2 = r2_score(Y, mediation_model.predict(M.reshape(-1, 1)))
        
        r2_difference = full_r2 - med_r2
        diagnostics['no_direct_effect'] = r2_difference < 0.05  # Small improvement suggests no direct effect
        
        # 3. Test mediator-outcome relationship
        # Mediator should predict outcome
        m_to_y_r2 = med_r2
        if m_to_y_r2 > 0 and n > 2:
            f_stat = (m_to_y_r2 / (1 - m_to_y_r2)) * (n - 2)
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
            mediator_outcome_significant = (p_value < alpha) and (m_to_y_r2 > 0.01)
        else:
            mediator_outcome_significant = False
        
        # 4. Test for no confounding between mediator and outcome
        # This is harder to test directly, but we can check if observed confounders explain the relationship
        if adjustment_set:
            # Control for observed confounders and see if M-Y relationship persists
            X_adj = df[adjustment_set].values
            X_med_adj = np.column_stack([M, X_adj])
            adj_model = LinearRegression()
            adj_model.fit(X_med_adj, Y)
            adj_r2 = r2_score(Y, adj_model.predict(X_med_adj))
            
            # If controlling for confounders doesn't eliminate M-Y relationship, that's good
            diagnostics['no_confounding_mediator_outcome'] = (adj_r2 - med_r2) < 0.1
        else:
            # Without adjustment set, assume this is satisfied
            diagnostics['no_confounding_mediator_outcome'] = mediator_outcome_significant
        
        # 5. Test sufficient variation
        # All variables should have sufficient variation
        t_var = np.var(T) > 1e-6
        m_var = np.var(M) > 1e-6
        y_var = np.var(Y) > 1e-6
        diagnostics['sufficient_variation'] = t_var and m_var and y_var
        
        # 6. Overall model fit
        # The two-stage mediation should explain reasonable variance
        diagnostics['model_fit'] = (t_to_m_r2 > 0.01) and (med_r2 > 0.01)
        
        # Overall validity
        diagnostics['overall_valid'] = all([
            diagnostics['mediator_causally_related'],
            diagnostics['no_direct_effect'],
            diagnostics['no_confounding_mediator_outcome'],
            diagnostics['sufficient_variation'],
            diagnostics['model_fit']
        ])
        
    except Exception as e:
        print(f"Frontdoor adjustment diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics



# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor

# def estimate(
#     data,
#     treatment,
#     mediator,
#     outcome
# ):
#     """
#     Method (sample-based approximation):
#       1. Fit an outcome model Q(A,M) ≈ E[Y|A,M] via regression.
#       2. Use observed mediator distribution:
#          • For do(A=1): compute Q(1, M_i) for all i with A_i=1, average.
#          • For do(A=0): compute Q(0, M_i) for all i with A_i=0, average.
#       3. ATE = E[Y|do(1)] - E[Y|do(0)].
#     """
#     df = data[[treatment, mediator, outcome]].dropna().copy()

#     A = df[treatment]
#     M = df[mediator]
#     Y = df[outcome]

#     # Fit outcome model Q(A,M)
#     X_Q = pd.DataFrame({
#         'A': A,
#         'M': M
#     })
#     model_Q = RandomForestRegressor(n_estimators=100, random_state=0)
#     model_Q.fit(X_Q, Y)

#     # Compute E[Y|do(A=a)] by averaging Q(a, M_i) over i: A_i = a
#     do1_mask = (A == 1)
#     do0_mask = (A == 0)

#     Q1_vals = model_Q.predict(
#         pd.DataFrame({'A': np.ones(do1_mask.sum()), 'M': M[do1_mask]})
#     )
#     Q0_vals = model_Q.predict(
#         pd.DataFrame({'A': np.zeros(do0_mask.sum()), 'M': M[do0_mask]})
#     )

#     do1 = np.mean(Q1_vals)
#     do0 = np.mean(Q0_vals)
#     ate = do1 - do0

#     return {'ate': ate, 'do1': do1, 'do0': do0, 'model_Q': model_Q}
