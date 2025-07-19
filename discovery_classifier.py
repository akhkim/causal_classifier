from collections import Counter
import numpy as np
import pandas as pd
from fitter import Fitter
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from discovery_algorithms import cgnn, fci, ges, pc, lingam, notears
import torch

def detect_likelihood(df):
    # If the dataframe is large, sample 100 rows at random
    if df.shape[0] > 100:
        df_sample = df.sample(n=100, random_state=0)
    else:
        df_sample = df

    cats, nums = [], []
    for col in df_sample.columns:
        if pd.api.types.is_numeric_dtype(df_sample[col]):
            nums.append(col)
        else:
            cats.append(col)

    winners, pvals = [], []

    # Continuous variables
    for col in nums:
        data = df_sample[col].dropna().values
        # Try fitting various distributions
        f = Fitter(data, distributions=["norm", "expon", "laplace"], timeout=30)
        f.fit()
        best_name = f.get_best(method="sumsquare_error").popitem()[0]

        if best_name == "norm":
            ad_stat, p_val = normal_ad(data)
        else:
            frozen_dist = getattr(stats, best_name)
            res = stats.goodness_of_fit(
                frozen_dist,
                data,
                statistic="ad",
                n_mc_samples=500
            )
            p_val = res.pvalue

        winners.append(best_name)
        pvals.append(p_val)

    # Discrete variables: label simply as "discrete" and p-value of 1
    for col in cats:
        winners.append("discrete")
        pvals.append(1.0)

    # Determine the most common family and check goodness-of-fit
    freq = Counter(winners)
    best_family, count = freq.most_common(1)[0]
    agree_ratio = count / len(winners)
    good_pvals = all(p > 0.05 for p in pvals)
    trust = (agree_ratio >= 0.8) and good_pvals

    return {
        "trust_likelihood": trust,
        "suggested_family": best_family if trust else None,
        "agreement_ratio": agree_ratio,
        "goodness_ok": good_pvals,  # all goodness-of-fit p-values > 0.05
        "per_variable": dict(zip(df_sample.columns, winners))
    }

def recommend_discovery_algorithm(df, latent_confounders):
    n, p = df.shape
    large_sample = n >= 5 * p
    distribution = detect_likelihood(df)["suggested_family"]

    if latent_confounders:
        return "FCI"  # Constraint-based search for latent confounders
    if distribution != "norm":
        if distribution == "discrete":
            return "GES"
        return "LiNGAM" if large_sample else "CGNN"
    if distribution == "norm":
        if large_sample:
            return "PC"
        return "NOTEARS"
    else:
        if torch.cuda.is_available():
            return "CGNN"  # Runs on minimal assumptions, and can handle non-linearity (most flexible on scarce data)
        return "NOTEARS"  # Use PC for small samples, as it is more robust to violations of assumptions

def run_discovery_algorithm(df, latent_confounders):
    algorithm = recommend_discovery_algorithm(df, latent_confounders)

    match algorithm:
        case "FCI":
            return fci.run(
                df,
                alpha=0.05,
                indep_test="fisherz"
            )
        case "PC":
            return pc.run(
                df,
                alpha=0.05,
                indep_test="fisherz"
            )
        case "GES":
            return ges.run(
                df,
                score_func="local_score_BIC"
            )
        case "LiNGAM":
            return lingam.run(
                df
            )
        case "NOTEARS":
            return notears.run(
                df,
                lambda1=0.01,
                max_iter=100,
                w_threshold=0.3
            )
        case "CGNN":
            return cgnn.run(
                df, 
                nh="auto", 
                nruns="auto"
            )
        case _:
            raise ValueError(f"Unknown discovery algorithm: {algorithm}")