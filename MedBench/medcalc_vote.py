"""MedCalc: connected-component plurality voting over the base models' answers.

Usage:
    python medcalc_vote.py

Reads data/filtered_medCalc.csv, writes results/vpa_results_abstain_ties.csv and
results/vpa_results_random_ties.csv (one mv_{k} column per plurality k).
"""
import os
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "data", "filtered_medCalc.csv")
RES  = os.path.join(HERE, "results")
META = ["question_id", "calculator_name", "output_type",
        "correct_answer", "lower_limit", "upper_limit"]
TOL  = {"decimal": 0.05}
KS   = range(1, 9)
SEED = 0


def _components(close):
    """Connected components of a boolean adjacency matrix. Returns list of index arrays."""
    n = close.shape[0]
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.flatnonzero(close[i] & ~seen):
                seen[j] = True
                stack.append(j)
        comps.append(np.array(comp))
    return comps


def vote_by_coverage(vals, tol=0.0, k=1, abs_tie=False, rng=None):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return None

    d     = np.abs(v[:, None] - v[None, :])
    scale = np.maximum(np.abs(v[:, None]), np.abs(v[None, :]))
    close = (d <= tol * scale) if tol else (d == 0)

    comps = _components(close)
    sizes = np.array([len(c) for c in comps])
    best  = int(sizes.max())
    if best < k:
        return None

    winners = [comps[i] for i in np.flatnonzero(sizes == best)]
    if len(winners) > 1:
        if abs_tie:
            return None
        rng = rng or random.Random(SEED)
        cluster = winners[rng.randrange(len(winners))]
    else:
        cluster = winners[0]

    members = v[cluster]
    return float(members[np.argmin(np.abs(members - np.median(members)))])


def build(df, resp, tols, abs_tie, dst):
    out = (df[["question_id", "correct_answer", "lower_limit",
               "upper_limit", "output_type"]]
           .rename(columns={"correct_answer": "exact_response"}))
    for k in KS:
        col = []
        for i, tol in zip(df.index, tols):
            rng = random.Random(f"{SEED}|{df.at[i, 'question_id']}|{k}")
            col.append(vote_by_coverage(resp.loc[i].dropna().tolist(), tol,
                                        k=k, abs_tie=abs_tie, rng=rng))
        out[f"mv_{k}"] = col
    out.to_csv(dst, index=False)
    return out


def report(out, label):
    LO, HI = out["lower_limit"].values, out["upper_limit"].values
    print(f"\n=== {label}")
    for k in KS:
        v   = out[f"mv_{k}"].values.astype(float)
        ans = ~np.isnan(v)
        ok  = (v >= LO) & (v <= HI)
        n   = ans.sum()
        acc = 100 * ok[ans].mean() if n else float("nan")
        print(f"k={k}  answered={n:3d}  coverage={100*n/len(out):5.1f}%  "
              f"acc={acc:5.1f}%  risk={100-acc:5.1f}%")


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    df   = pd.read_csv(SRC)
    mods = [c for c in df.columns if c not in META]
    resp = df[mods].apply(pd.to_numeric, errors="coerce")
    tols = [TOL.get(str(t).strip().lower(), 0.0) for t in df["output_type"]]

    vpa_abstain = build(df, resp, tols, True,  os.path.join(RES, "vpa_results_abstain_ties.csv"))
    vpa_random  = build(df, resp, tols, False, os.path.join(RES, "vpa_results_random_ties.csv"))

    report(vpa_abstain, "abstain on ties")
    report(vpa_random,  "random tie-break")
