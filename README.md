# Making MTM Savings Defensible

**Regression to the mean, engagement selection, and a matched difference-in-differences estimator for medication-optimization programs.**

Medication therapy management (MTM) and medication-optimization programs usually report outcomes as *"$X saved per engaged member per year"* — a pre/post comparison on the engaged, high-risk subgroup. Once a contract moves to value-based terms, the client's actuary asks how much of that would have happened anyway.

This repo shows, on **fully simulated** Medicare-like data:

| True program effect | Naive pre/post (engaged members) | Propensity-matched DiD |
|---|---|---|
| **0%** | **−35%** (~$28K "saved" PMPY) | −2% (CI includes 0) |
| **−10%** | −41% (~$33K PMPY) | **−12%** (CI −15% to −9%) |

The naive number barely changes when the program goes from useless to genuinely effective — it is measuring selection, not the program. The matched DiD lands on the truth in both cases and across 30 repeated simulations.

## Contents

- `mtm_savings_evaluation.ipynb` — the walkthrough with figures (also exported as `.html`)
- `mtm_eval.py` — simulation, naive estimator, propensity matching, balance table, DiD, Monte Carlo
- `MEMO.md` — one-page summary for a non-technical finance / contracting audience

## Run it

```bash
pip install -r requirements.txt
python mtm_eval.py            # prints the headline comparison
jupyter notebook mtm_savings_evaluation.ipynb
```

## Method in one paragraph

Members are targeted on a pre-period risk score (top 10%); engagement among targeted members depends on medication count, frailty and dual status. Engaged members are matched 1:1 (nearest neighbour, caliper 0.05) to non-engaged members on a propensity score that **includes log pre-period cost**, which neutralises regression to the mean. The effect is estimated as a difference-in-differences on log cost with standard errors clustered by matched pair. Balance is reported as standardised mean differences (target |SMD| < 0.10).

## Pre-reporting checklist

1. Comparison group selected by the same targeting rule — never "engaged vs. everyone else"
2. Balance diagnostics on risk score, prior cost, prior utilisation, chronic burden, dual status
3. Difference-in-differences with a confidence interval, on cost and on admits / ED visits
4. Placebo test on a pre-program year pair
5. Sensitivity to caliper and covariates; an intention-to-treat estimate on all *targeted* members
6. A conservative one-pager for the client's finance team

*All data is simulated. No PHI or client data is used.*

— Faridoon Farahi · [linkedin.com/in/faridoon-farahi](https://linkedin.com/in/faridoon-farahi)
