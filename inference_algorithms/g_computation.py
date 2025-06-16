from dowhy import CausalModel

def estimate(
    data,
    treatment,
    outcome,
    adjustment_set
):
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=adjustment_set
    )
    identified_estimand = model.identify_effect()
    est = model.estimate_effect(
        identified_estimand,
        method_name="gformula",
        method_params={"n_simulations": 1000}
    )
    return {
        'estimate': est.value,
        'std_error': est.get_std_error(),
        'model': est
    }


# import numpy as np
# from sklearn.base import clone
# from sklearn.linear_model import LinearRegression

# def estimate(
#     data,
#     treatment,
#     outcome,
#     adjustment_set,
#     effect='ate'
# ):
#     df = data.copy()
#     outcome_model = LinearRegression()

#     # Prepare feature matrix
#     features = adjustment_set + [treatment]
#     X = df[features]
#     Y = df[outcome]

#     # Fit the outcome regression model
#     model = clone(outcome_model).fit(X, Y)

#     # Create data for potential outcomes
#     df1 = df.copy()
#     df1[treatment] = 1
#     X1 = df1[features]

#     df0 = df.copy()
#     df0[treatment] = 0
#     X0 = df0[features]

#     # Predict potential outcomes
#     Y1_pred = model.predict(X1)
#     Y0_pred = model.predict(X0)

#     # Individual treatment effects
#     ITE = Y1_pred - Y0_pred

#     # Compute effect estimate
#     if effect.lower() == 'ate':
#         estimate = np.mean(ITE)
#     elif effect.lower() == 'att':
#         mask = df[treatment] == 1
#         estimate = np.mean(ITE[mask])

#     return {
#         'estimate': estimate,
#         'std_error': None,
#         'model': model
#     }

