from dowhy import CausalModel

def estimate(
    data,
    treatment,
    mediator,
    adjustment_set,
    outcome
):
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
