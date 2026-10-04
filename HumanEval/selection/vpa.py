"""Our selection method (plurality vote over similarity groups, with abstention) on HumanEval.

Usage:
    python selection/vpa.py

Runs the settings used in the paper for every similarity (tests, init_tests, crosshair, codebleu):
tau in 2, 3, 6, 7 with and without ranking by tests passed, random tie-break or abstain on ties.

Reads  data/adj/{metric}_sim.npz, data/all_completions.jsonl, data/{rbased,init_test}_matrix.json
Writes results/mv_out/max_completions/{stem}.jsonl   selected solution per task ("ABS" = abstain)
       results/mv_out/max_rec_stats/{stem}_records.json
"""
import argparse
import json, os, random
import numpy as np
from collections import defaultdict, Counter

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
ADJ    = os.path.join(BASE, "data", "adj")
OUTDIR = os.path.join(BASE, "results", "mv_out")
for sub in ("", "max_completions", "max_rec_stats"):
    os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)

COMPLETIONS = os.path.join(BASE, "data", "all_completions.jsonl")
FP_FILE     = os.path.join(BASE, "data", "rbased_matrix.json")
IT_FILE     = os.path.join(BASE, "data", "init_test_matrix.json")
ABS         = "ABS"

TAU_DEFAULT = {"crosshair": 1.0, "codebleu": 0.9, "tests": 1.0, "init_tests": 1.0}
NT_FILE     = {"init_tests": IT_FILE}          # everything else -> FP_FILE


# loading
def load_metric(metric):
    d = np.load(os.path.join(ADJ, f"{metric}_sim.npz"))
    return dict(sim=d["sim"], known=d["known"], valid=d["valid"], passed=d["passed"],
                tasks=list(d["tasks"]), models=[str(m).strip() for m in d["models"]])


def load_ntests(metric="tests"):
    doc = json.load(open(NT_FILE.get(metric, FP_FILE), encoding="utf-8"))
    out = {}
    for tid, rows in doc["matrix"].items():
        lens = {len(v) for v in rows.values() if v is not None}
        out[tid] = lens.pop() if len(lens) == 1 else max(lens, default=0)
    return out


def load_solutions(path):
    out = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                out[e["task_id"]][(e.get("model") or "").strip()] = e
    return out



def to_adjacency(sim, known, valid, tau, require_both=True):
    e   = (sim >= tau) & known
    adj = (e & e.T) if require_both else (e | e.T)
    adj &= valid[:, None] & valid[None, :]
    np.fill_diagonal(adj, valid)
    return adj


def components(adj):
    seen, out = set(), []
    for i in range(len(adj)):
        if i in seen or not adj[i, i]:
            continue
        comp, stack = [], [i]; seen.add(i)
        while stack:
            c = stack.pop(); comp.append(c)
            for j in np.flatnonzero(adj[c]):
                if int(j) not in seen:
                    seen.add(int(j)); stack.append(int(j))
        out.append(sorted(comp))
    return out



def rank_components(comps, passed, k, passed_tests):
    elig = [c for c in comps if len(c) >= k]
    if not elig:
        return [], []
    if passed_tests:
        key = lambda c: (max(int(passed[i]) for i in c), len(c))
    else:
        key = lambda c: (len(c),)
    groups = defaultdict(list)
    for c in elig:
        groups[key(c)].append(c)
    return [(s, groups[s]) for s in sorted(groups, reverse=True)], elig


def draw(tied, rng, P=None):
    comp = tied[rng.randrange(len(tied))]
    if P is None:
        return comp, comp[rng.randrange(len(comp))]
    best  = max(int(P[i]) for i in comp)
    cands = [i for i in comp if int(P[i]) == best]
    return comp, cands[rng.randrange(len(cands))]


def make_entry(comp, rep, score, n_tied, P, models, rank):
    return {
        "rank":            rank,
        "picked_model":    models[rep],
        "component_size":  len(comp),
        "supporters":      [models[i] for i in comp],
        "score":           list(score),
        "n_tied_at_score": n_tied,
        "picked_tests_passed":    int(P[rep]),
        "component_tests_passed": [int(P[i]) for i in comp],
    }



def run(metric, k, passed_tests=False, abs_ties=False, tau=None, seed=0,
        require_both=True, tag="",
        M=None, SOL=None, NT=None):

    tau = TAU_DEFAULT[metric] if tau is None else tau
    M   = M   if M   is not None else load_metric(metric)
    SOL = SOL if SOL is not None else load_solutions(COMPLETIONS)
    NT  = NT  if NT  is not None else load_ntests(metric)
    S, K, V, P = M["sim"], M["known"], M["valid"], M["passed"]
    tasks, models = M["tasks"], M["models"]

    stem = (f"{metric}_k{k}_tau{tau}_pt{int(passed_tests)}"
            f"_ties{int(abs_ties)}_s{seed}{tag}")
    f_jsonl  = os.path.join(OUTDIR, "max_completions", f"{stem}.jsonl")
    f_recs   = os.path.join(OUTDIR, "max_rec_stats",  f"{stem}_records.json")


    records, abstained_ids = {}, []

    with open(f_jsonl, "w", encoding="utf-8") as fh:
        for t, tid in enumerate(tasks):
            rng   = random.Random(f"{seed}|{tid}")
            adj   = to_adjacency(S[t], K[t], V[t], tau, require_both)
            comps = components(adj)
            ranked, elig = rank_components(comps, P[t], k, passed_tests)

            valid_p = P[t][V[t]]
            base = {
                "metric": metric, "tau": tau, "k": k,
                "passed_tests": passed_tests, "abs_ties": abs_ties,
                "seed": seed, "require_both": require_both,
                "n_tests":          int(NT.get(tid, 0)),
                "n_valid":          int(V[t].sum()),
                "n_components":     len(comps),
                "component_sizes":  sorted((len(c) for c in comps), reverse=True),
                "n_eligible_comps": len(elig),
                "max_passed_any":   int(valid_p.max()) if valid_p.size else 0,
            }

            if not ranked:
                records[tid] = {**base, "abstained": True, "reason": "below_k",
                                "picked_model": None, "component_size": 0,
                                "supporters": [], "picked_tests_passed": None,
                                "component_tests_passed": [],
                                "n_tied_at_top": 0, "n_tied_on_tests": None,
                                "margin": None, "picked_is_best": None}
                
                abstained_ids.append(tid)
                fh.write(json.dumps({"task_id": tid, "solution": ABS}) + "\n")
                continue

            top_score, top_tied = ranked[0]
            n_tied_tests = (sum(len(g) for s, g in ranked if s[0] == top_score[0])
                            if passed_tests else None)

            if len(top_tied) > 1 and abs_ties:
                records[tid] = {**base, "abstained": True, "reason": "tied",
                                "picked_model": None, "component_size": 0,
                                "supporters": [], "picked_tests_passed": None,
                                "component_tests_passed": [],
                                "n_tied_at_top": len(top_tied),
                                "n_tied_on_tests": n_tied_tests,
                                "margin": None, "picked_is_best": None}
                
                abstained_ids.append(tid)
                fh.write(json.dumps({"task_id": tid, "solution": ABS}) + "\n")
                continue

            Pd = P[t] if passed_tests else None
            comp, rep = draw(top_tied, rng, Pd)

  

            name = models[rep]
            sol  = SOL.get(tid, {}).get(name, {}).get("solution") or ""

            records[tid] = {
                **base, "abstained": False, "reason": "ok",
                "picked_model":   name,
                "component_size": len(comp),
                "supporters":     [models[i] for i in comp],
                "picked_tests_passed":    int(P[t][rep]),
                "component_tests_passed": [int(P[t][i]) for i in comp],
                "n_tied_at_top":   len(top_tied),
                "n_tied_on_tests": n_tied_tests,
                "picked_is_best": int(P[t][rep]) == base["max_passed_any"],
            }
            
            fh.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")

    json.dump(records, open(f_recs,   "w", encoding="utf-8"), indent=1)
    

    summarise(records, abstained_ids, stem)
    return records, f_jsonl, f_recs



def summarise(recs, abstained_ids, stem="", verbose=True):
    n    = len(recs)
    ans  = [r for r in recs.values() if not r["abstained"]]
    reas = Counter(r["reason"] for r in recs.values() if r["abstained"])
    any0 = [r for r in ans if r["component_tests_passed"]
            and max(r["component_tests_passed"]) == 0]
    best  = [r for r in ans if r["picked_is_best"]]
    split = [r for r in ans if len(set(r["component_tests_passed"])) > 1]
    one   = next(iter(recs.values()))

    row = {
        "stem": stem, "metric": one["metric"], "tau": one["tau"],
        "k": one["k"], "passed_tests": one["passed_tests"],
        "abs_ties": one["abs_ties"], "seed": one["seed"],
        "n_tasks": n, "n_answered": len(ans), "n_abstained": n - len(ans),
        "coverage": round(len(ans) / n, 4) if n else 0.0,
        "n_below_k": reas.get("below_k", 0), "n_tied": reas.get("tied", 0),
        "mean_comp_size": round(float(np.mean([r["component_size"] for r in ans])), 3)
                          if ans else 0.0,
        "n_zero_test_wins": len(any0),
        "frac_picked_is_best": round(len(best) / len(ans), 4) if ans else None,
        "n_internal_disagree": len(split),
        "abstained_task_ids": ";".join(abstained_ids),
    }

    if verbose:
        print(f"\n=== {stem} ===")
        print(f"answered {len(ans)}/{n}  coverage {row['coverage']:.3f}  "
              f"below_k {row['n_below_k']}  tied {row['n_tied']}")
        if ans:
            sizes = Counter(r["component_size"] for r in ans)
            print("  sizes: " + " ".join(f"{s}:{sizes[s]}" for s in sorted(sizes)))
            print(f"  picked==best-available: {row['frac_picked_is_best']:.3f} | "
                  f"zero-test wins: {len(any0)} | "
                  f"internal disagreement: {len(split)}")
    return row





def sweep(metrics=("crosshair", "codebleu", "tests", "init_tests"),
          ks=(2,3,6,7), seeds=(0,), pts=(False,), ats=(False,True),
          taus=None, tag=""):
    SOL = load_solutions(COMPLETIONS)
    NTS = {m: load_ntests(m) for m in set(metrics)}
    for metric in metrics:
        M = load_metric(metric)
        for tau in (taus or {}).get(metric, [None]):
            for k in ks:
                for pt in pts:
                    for at in ats:
                        for s in seeds:
                            run(metric, k=k, passed_tests=pt, abs_ties=at,
                                tau=tau, seed=s, M=M, SOL=SOL,
                                NT=NTS[metric], tag=tag)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()   # only -h
    # the runs used by the paper figures
    ALL = ("tests", "init_tests", "crosshair", "codebleu")
    sweep(metrics=ALL, ks=(2, 3, 6, 7), pts=(True, False), ats=(False, True))
    sweep(metrics=ALL, ks=(1,), pts=(True,), ats=(False,))



