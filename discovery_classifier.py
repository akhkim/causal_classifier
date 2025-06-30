from collections import Counter
import numpy as np
import pandas as pd
from fitter import Fitter
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from discovery_algorithms import cgnn, fci, ges, pc, lingam, notears
import torch

def _chi2_goodness_of_fit(series, dist_name, params):
    x = series.values
    values, counts = np.unique(x, return_counts=True)
    n = len(series)
    dist = getattr(stats, dist_name)
    if dist_name == "binom":
        k, p = params
        expected = dist.pmf(values, k, p) * n
    else:
        expected = dist.pmf(values, *params) * n
    expected = np.maximum(expected, 1e-9)
    chi2 = ((counts - expected) ** 2 / expected).sum()
    df_chi = max(len(values) - 1 - len(params), 1)
    p_val = 1 - stats.chi2.cdf(chi2, df_chi)
    return p_val

def detect_likelihood(df):
    cats, nums = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            nums.append(col)
        else:
            cats.append(col)

    winners, pvals = [], []

    # Continuous variables
    for col in nums:
        data = df[col].dropna().values
        f = Fitter(data, distributions=["norm", "gamma", "expon", "laplace"], timeout=30)
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

    # Discrete variables
    for col in cats:
        winners.append("discrete")
        pvals.append(1)

    freq = Counter(winners)
    best_family, count = freq.most_common(1)[0]
    agree_ratio = count / len(winners)
    good_pvals = all(p > 0.05 for p in pvals)

    trust = (agree_ratio >= 0.8) and good_pvals

    return {
        "trust_likelihood": trust,
        "suggested_family": best_family if trust else None,
        "agreement_ratio": agree_ratio,
        "goodness_ok": good_pvals,    # Goodness of fit p-values
        "per_variable": dict(zip(df.columns, winners)),
    }

def recommend_discovery_algorithm(df, latent_confounders):
    n, p = df.shape
    large_sample = n >= 5 * p
    distribution = detect_likelihood(df)["suggested_family"]

    if latent_confounders:
        return "FCI"  # Constraint-based search for latent confounders
    if distribution == "discrete":
        return "GES"
    elif distribution != "norm":
        return "LiNGAM"
    elif distribution == "norm":
        if large_sample:
            return "GES"
        return "NOTEARS"
    else:
        if torch.cuda.is_available():
            return "CGNN"  # Runs on minimal assumptions, and can handle non-linearity (most flexible on scarce data)
        return "PC"  # Use PC for small samples, as it is more robust to violations of assumptions

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
                df,
                max_iter=1000
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