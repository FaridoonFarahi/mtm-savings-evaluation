"""
Making MTM savings defensible: regression to the mean, engagement selection,
and a matched difference-in-differences estimator.

All data here is SIMULATED. The point is not the numbers; it is that a naive
pre/post comparison on engaged high-risk members produces large "savings"
even when the program does nothing, and that a propensity-matched
difference-in-differences design recovers the true effect.

Author: Faridoon Farahi
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------------
# 1. Simulate a Medicare-like plan population
# ----------------------------------------------------------------------------
def simulate_population(
    n: int = 80_000,
    true_effect: float = -0.10,   # program lowers post-period cost by 10% for treated
    seed: int = 42,
) -> pd.DataFrame:
    """
    Two 12-month periods (pre, post). Each member has a persistent underlying
    risk (chronic burden) plus a large transient shock each period. Annual cost
    is heavy-tailed (log-normal), which is what real claims look like.

    Program targeting: the plan identifies members using a risk score that is
    computed on PRE-period utilization. Members who look expensive in the pre
    period are targeted. Only some targeted members engage, and engagement
    itself is correlated with member characteristics.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(72, 8, n).clip(65, 95)
    n_chronic = rng.poisson(2.2, n).clip(0, 9)
    n_meds = (n_chronic * 2 + rng.poisson(3, n)).clip(0, 25)
    dual_eligible = rng.random(n) < 0.25
    frailty = rng.normal(0, 1, n)

    # Persistent latent risk (what a good risk model would try to capture)
    latent = (
        0.02 * (age - 72)
        + 0.25 * n_chronic
        + 0.05 * n_meds
        + 0.3 * dual_eligible
        + 0.3 * frailty
    )

    # Transient shocks: the source of regression to the mean
    shock_pre = rng.normal(0, 0.6, n)
    shock_post = rng.normal(0, 0.6, n)

    base_log_cost = 8.6  # exp(8.6) ~ $5,400 median annual cost
    cost_pre = np.exp(base_log_cost + latent + shock_pre)

    # Plan targets members whose PRE cost lands in the top decile (a common
    # "high-risk" rule). Risk score = pre cost + noise from an imperfect model.
    risk_score = np.log(cost_pre) + rng.normal(0, 0.2, n)
    targeted = risk_score >= np.quantile(risk_score, 0.90)

    # Engagement among targeted members is not random: more meds, lower
    # frailty and non-dual members engage more (they answer the phone).
    engage_logit = -0.4 + 0.06 * n_meds - 0.35 * frailty - 0.3 * dual_eligible
    engaged = targeted & (rng.random(n) < 1 / (1 + np.exp(-engage_logit)))

    # Post-period cost: same latent risk, NEW shock, plus true effect for engaged
    effect = np.where(engaged, np.log1p(true_effect), 0.0)
    cost_post = np.exp(base_log_cost + latent + shock_post + effect)

    df = pd.DataFrame(
        dict(
            member_id=np.arange(n),
            age=age.round(0),
            n_chronic=n_chronic,
            n_meds=n_meds,
            dual_eligible=dual_eligible.astype(int),
            frailty=frailty.round(3),
            risk_score=risk_score.round(3),
            targeted=targeted.astype(int),
            engaged=engaged.astype(int),
            cost_pre=cost_pre.round(2),
            cost_post=cost_post.round(2),
        )
    )
    df["delta"] = df["cost_post"] - df["cost_pre"]
    return df


# ----------------------------------------------------------------------------
# 2. The naive estimator everyone reports
# ----------------------------------------------------------------------------
def naive_pre_post(df: pd.DataFrame) -> dict:
    """Mean cost change among ENGAGED members only. This is what
    'savings per engaged member per year' usually means."""
    e = df[df.engaged == 1]
    pre, post = e.cost_pre.mean(), e.cost_post.mean()
    return dict(
        n_engaged=len(e),
        mean_cost_pre=pre,
        mean_cost_post=post,
        savings_pmpy=pre - post,
        pct_change=(post - pre) / pre,
    )


# ----------------------------------------------------------------------------
# 3. Propensity-matched difference-in-differences
# ----------------------------------------------------------------------------
def propensity_match(
    df: pd.DataFrame,
    covariates: list[str],
    caliper: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    1:1 nearest-neighbour matching on the propensity to engage, with a caliper.
    Comparison pool = members who were NOT engaged. Because pre-period cost is
    in the covariate list, matched controls share the same 'looked expensive
    last year' property, which is what neutralises regression to the mean.
    """
    X = df[covariates].copy()
    X["log_cost_pre"] = np.log(df.cost_pre)
    X = (X - X.mean()) / X.std()
    y = df.engaged.values

    ps = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    df = df.assign(pscore=ps)

    treated = df[df.engaged == 1]
    control = df[df.engaged == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control[["pscore"]].values)
    dist, idx = nn.kneighbors(treated[["pscore"]].values)
    keep = dist.ravel() <= caliper

    matched_t = treated[keep].assign(pair=np.arange(keep.sum()))
    matched_c = control.iloc[idx.ravel()[keep]].assign(pair=np.arange(keep.sum()))
    return pd.concat([matched_t, matched_c], ignore_index=True)


def balance_table(matched: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Standardised mean differences before/after matching. |SMD| < 0.1 is the
    usual 'balanced' threshold an actuary or HEOR reviewer will look for."""
    rows = []
    for c in covariates + ["cost_pre"]:
        t = matched.loc[matched.engaged == 1, c]
        k = matched.loc[matched.engaged == 0, c]
        pooled_sd = np.sqrt((t.var() + k.var()) / 2)
        rows.append(dict(covariate=c, treated_mean=t.mean(), control_mean=k.mean(),
                         smd=(t.mean() - k.mean()) / pooled_sd))
    return pd.DataFrame(rows)


def did_estimate(matched: pd.DataFrame) -> dict:
    """
    Difference-in-differences on log cost. With two periods and member fixed
    effects this is identical to regressing each member's pre->post change in
    log cost on the treatment indicator, so we estimate it that way (no
    rank-deficiency from thousands of member dummies). Cluster-robust SEs by
    matched pair.
    """
    m = matched.assign(dlog=np.log(matched.cost_post) - np.log(matched.cost_pre))
    model = smf.ols("dlog ~ engaged", data=m).fit(
        cov_type="cluster", cov_kwds={"groups": m.pair}
    )
    coef = model.params["engaged"]
    se = model.bse["engaged"]
    ci = model.conf_int().loc["engaged"]
    return dict(
        did_log=coef,
        did_pct=np.expm1(coef),
        ci_pct=(np.expm1(ci[0]), np.expm1(ci[1])),
        se=se,
        n_pairs=matched.pair.nunique(),
    )


def dollar_savings_pmpy(matched: pd.DataFrame, did_pct: float) -> float:
    """Translate the DiD % effect into dollars per engaged member per year
    using the matched-control post-period mean as the counterfactual."""
    counterfactual = matched.loc[matched.engaged == 0, "cost_post"].mean()
    return -did_pct * counterfactual


# ----------------------------------------------------------------------------
# 4. Placebo: run the whole pipeline on a world with ZERO true effect
# ----------------------------------------------------------------------------
def run_pipeline(true_effect: float, covariates: list[str], seed: int = 42) -> dict:
    df = simulate_population(true_effect=true_effect, seed=seed)
    naive = naive_pre_post(df)
    matched = propensity_match(df, covariates)
    did = did_estimate(matched)
    return dict(
        true_effect=true_effect,
        naive_pct=naive["pct_change"],
        naive_savings_pmpy=naive["savings_pmpy"],
        did_pct=did["did_pct"],
        did_ci=did["ci_pct"],
        did_savings_pmpy=dollar_savings_pmpy(matched, did["did_pct"]),
        n_engaged=naive["n_engaged"],
        n_pairs=did["n_pairs"],
    )


COVARIATES = ["age", "n_chronic", "n_meds", "dual_eligible", "frailty"]


def monte_carlo(true_effect: float, covariates: list[str], n_sims: int = 30, n: int = 30_000) -> pd.DataFrame:
    """Repeat the whole pipeline over many simulated populations to show the
    naive estimator is biased and the matched DiD is centred on the truth."""
    rows = []
    for s in range(n_sims):
        df = simulate_population(n=n, true_effect=true_effect, seed=1000 + s)
        naive = naive_pre_post(df)["pct_change"]
        did = did_estimate(propensity_match(df, covariates))["did_pct"]
        rows.append(dict(sim=s, naive_pct=naive, did_pct=did))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for te in (0.0, -0.10):
        r = run_pipeline(te, COVARIATES)
        print(f"\nTrue effect: {te:+.0%}")
        print(f"  Naive pre/post on engaged:  {r['naive_pct']:+.1%}  (${r['naive_savings_pmpy']:,.0f} PMPY)")
        print(f"  Matched DiD:                {r['did_pct']:+.1%}  CI [{r['did_ci'][0]:+.1%}, {r['did_ci'][1]:+.1%}]  (${r['did_savings_pmpy']:,.0f} PMPY)")
