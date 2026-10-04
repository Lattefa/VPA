"""Consensus selection under an adversary.

Adversary (per task):
    1. build the clean clusters (same graph rules as the selection script)
    2. target = second-largest cluster by size (random among equal sizes)
    3. victim = random member of the target
    4. add N copies of the victim. A copy inherits everything from the victim:
       similarity to every model, known/valid flags, number of tests passed.
       Copies and victim are identical code -> similarity 1 between them.
Then selection runs exactly as in the selection script (k filter, ranking, ties, draw).
A picked copy is written out as the victim's solution, so EvalPlus scores it as usual.

Usage:
    python selection/byzantine_he.py                                # tests, 1 copy, pt1, abstain on ties
    python selection/byzantine_he.py --metrics tests --copies 1,2,3,4 --ks 2,3,6,7
    python selection/byzantine_he.py --metrics tests,init_tests,crosshair,codebleu --pt 0,1 --ties 0,1

Writes adv_out/ (same layout and file format as mv_out/):
    max_completions/{metric}_k{k}_tau{tau}_pt{pt}_ties{ties}_s{seed}{tag}_adv{N}.jsonl
    max_rec_stats/{stem}_records.json
    max_runner_up/{stem}_runnerup.json
"""
import json, os, random, argparse
import numpy as np
from collections import defaultdict, Counter

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
ADJ    = os.path.join(BASE, "data", "adj")
OUTDIR = os.path.join(BASE, "results", "adv_out")
for sub in ("", "max_completions", "max_rec_stats", "max_runner_up"):
    os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)

COMPLETIONS = os.path.join(BASE, "data", "all_completions.jsonl")
FP_FILE     = os.path.join(BASE, "data", "rbased_matrix.json")
IT_FILE     = os.path.join(BASE, "data", "init_test_matrix.json")
ABS         = "ABS"

TAU_DEFAULT = {"crosshair": 1.0, "codebleu": 0.9, "tests": 1.0, "init_tests": 1.0}
NT_FILE     = {"init_tests": IT_FILE}          # everything else -> FP_FILE


# ------------------------------------------------------------------ loading (unchanged)
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


# graph
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


# ranking
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


#  adversary
def pick_target(comps, rng):
    """Second-largest cluster by size (random order among equal sizes) + random member."""
    if len(comps) < 2:
        return None, None
    order  = sorted(comps, key=lambda c: (len(c), rng.random()), reverse=True)
    target = order[1]
    return target, target[rng.randrange(len(target))]


def add_copies(S, K, V, P, victim, n_copies):
    """Append n_copies of `victim`. Returns augmented S, K, V, P and new->original index map."""
    N   = len(V)
    idx = list(range(N)) + [victim] * n_copies
    ix  = np.ix_(idx, idx)
    S2, K2 = S[ix].copy(), K[ix].copy()
    V2, P2 = V[idx].copy(), P[idx].copy()
    same = [victim] + list(range(N, N + n_copies))          # identical code
    for a in same:
        for b in same:
            S2[a, b], K2[a, b] = 1.0, True
    return S2, K2, V2, P2, idx



def run(metric, k, n_copies, passed_tests=True, abs_ties=True, tau=None, seed=0,
        require_both=True, tag="",
        M=None, SOL=None, NT=None):

    tau = TAU_DEFAULT[metric] if tau is None else tau
    M   = M   if M   is not None else load_metric(metric)
    SOL = SOL if SOL is not None else load_solutions(COMPLETIONS)
    NT  = NT  if NT  is not None else load_ntests(metric)
    S_all, K_all, V_all, P_all = M["sim"], M["known"], M["valid"], M["passed"]
    tasks, models = M["tasks"], M["models"]
    N = len(models)

    stem = (f"{metric}_k{k}_tau{tau}_pt{int(passed_tests)}"
            f"_ties{int(abs_ties)}_s{seed}{tag}_adv{n_copies}")
    f_jsonl  = os.path.join(OUTDIR, "max_completions", f"{stem}.jsonl")
    f_recs   = os.path.join(OUTDIR, "max_rec_stats",  f"{stem}_records.json")
    f_runner = os.path.join(OUTDIR, "max_runner_up",  f"{stem}_runnerup.json")

    records, runners, abstained_ids = {}, {}, []

    with open(f_jsonl, "w", encoding="utf-8") as fh:
        for t, tid in enumerate(tasks):
            rng = random.Random(f"{seed}|{tid}")                 # same as selection script
            S, K, V, P = S_all[t], K_all[t], V_all[t], P_all[t]

            #  adversary acts on the clean clusters
            clean = components(to_adjacency(S, K, V, tau, require_both))
            target, victim = pick_target(clean, random.Random(f"adv|{seed}|{tid}"))
            if victim is not None and n_copies > 0:
                S, K, V, P, idx = add_copies(S, K, V, P, victim, n_copies)
            else:
                idx = list(range(N))
            names = [models[i] if j < N else f"adv{j - N + 1}<{models[i]}>"
                     for j, i in enumerate(idx)]
            target_set = set(target or [])
            adv_info = {
                "n_copies":     n_copies if victim is not None else 0,
                "adv_target":   [models[i] for i in sorted(target_set)],
                "adv_victim":   models[victim] if victim is not None else None,
            }

            # selection, unchanged
            adj   = to_adjacency(S, K, V, tau, require_both)
            comps = components(adj)
            ranked, elig = rank_components(comps, P, k, passed_tests)

            valid_p = P[V]
            base = {
                "metric": metric, "tau": tau, "k": k,
                "passed_tests": passed_tests, "abs_ties": abs_ties,
                "seed": seed, "require_both": require_both,
                "n_tests":          int(NT.get(tid, 0)),
                "n_valid":          int(V.sum()),
                "n_components":     len(comps),
                "component_sizes":  sorted((len(c) for c in comps), reverse=True),
                "n_eligible_comps": len(elig),
                "max_passed_any":   int(valid_p.max()) if valid_p.size else 0,
                **adv_info,
            }
            empty = {"picked_model": None, "picked_source_model": None,
                     "picked_is_adversary": None, "picked_from_target": None,
                     "component_size": 0, "supporters": [],
                     "picked_tests_passed": None, "component_tests_passed": [],
                     "margin": None, "picked_is_best": None}

            if not ranked:
                records[tid] = {**base, **empty, "abstained": True, "reason": "below_k",
                                "n_tied_at_top": 0, "n_tied_on_tests": None}
                runners[tid] = None
                abstained_ids.append(tid)
                fh.write(json.dumps({"task_id": tid, "solution": ABS}) + "\n")
                continue

            top_score, top_tied = ranked[0]
            n_tied_tests = (sum(len(g) for s, g in ranked if s[0] == top_score[0])
                            if passed_tests else None)

            if len(top_tied) > 1 and abs_ties:
                records[tid] = {**base, **empty, "abstained": True, "reason": "tied",
                                "n_tied_at_top": len(top_tied),
                                "n_tied_on_tests": n_tied_tests}
                runners[tid] = None
                abstained_ids.append(tid)
                fh.write(json.dumps({"task_id": tid, "solution": ABS}) + "\n")
                continue

            Pd = P if passed_tests else None
            comp, rep = draw(top_tied, rng, Pd)

            e1 = None
            if len(ranked) > 1:
                s1, tied1 = ranked[1]
                c1, r1 = draw(tied1, rng, Pd)
                e1 = make_entry(c1, r1, s1, len(tied1), P, names, 1)
            elif len(top_tied) > 1:
                rest = [c for c in top_tied if c is not comp]
                c1, r1 = draw(rest, rng, Pd)
                e1 = make_entry(c1, r1, top_score, len(top_tied), P, names, 1)

            src = models[idx[rep]]                              # copy -> victim's model
            sol = SOL.get(tid, {}).get(src, {}).get("solution") or ""

            records[tid] = {
                **base, "abstained": False, "reason": "ok",
                "picked_model":        names[rep],
                "picked_source_model": src,
                "picked_is_adversary": rep >= N,
                "picked_from_target":  idx[rep] in target_set,
                "component_size": len(comp),
                "supporters":     [names[i] for i in comp],
                "picked_tests_passed":    int(P[rep]),
                "component_tests_passed": [int(P[i]) for i in comp],
                "n_tied_at_top":   len(top_tied),
                "n_tied_on_tests": n_tied_tests,
                "margin": ([int(a - b) for a, b in zip(top_score, e1["score"])]
                           if e1 else None),
                "picked_is_best": int(P[rep]) == base["max_passed_any"],
            }
            runners[tid] = e1
            fh.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")

    json.dump(records, open(f_recs,   "w", encoding="utf-8"), indent=1)
    json.dump(runners, open(f_runner, "w", encoding="utf-8"), indent=1)

    summarise(records, abstained_ids, stem)
    return records, f_jsonl, f_recs


# stats
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
        "n_copies": max(r["n_copies"] for r in recs.values()),
        "n_tasks": n, "n_answered": len(ans), "n_abstained": n - len(ans),
        "coverage": round(len(ans) / n, 4) if n else 0.0,
        "n_below_k": reas.get("below_k", 0), "n_tied": reas.get("tied", 0),
        "n_attacked": sum(r["adv_victim"] is not None for r in recs.values()),
        "n_picked_target":    sum(bool(r["picked_from_target"]) for r in ans),
        "n_picked_adversary": sum(bool(r["picked_is_adversary"]) for r in ans),
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
              f"below_k {row['n_below_k']}  tied {row['n_tied']}  "
              f"attacked {row['n_attacked']}  target won {row['n_picked_target']}")
        if ans:
            sizes = Counter(r["component_size"] for r in ans)
            print("  sizes: " + " ".join(f"{s}:{sizes[s]}" for s in sorted(sizes)))
            print(f"  picked==best-available: {row['frac_picked_is_best']:.3f} | "
                  f"zero-test wins: {len(any0)} | "
                  f"internal disagreement: {len(split)}")
    return row


def sweep(metrics=("tests",), copies=(1,), ks=(2, 3, 6, 7), seeds=(0,),
          pts=(True,), ats=(True,), taus=None, tag=""):
    SOL = load_solutions(COMPLETIONS)
    NTS = {m: load_ntests(m) for m in set(metrics)}
    for metric in metrics:
        M = load_metric(metric)
        for tau in (taus or {}).get(metric, [None]):
            for n in copies:
                for k in ks:
                    for pt in pts:
                        for at in ats:
                            for s in seeds:
                                run(metric, k=k, n_copies=n, passed_tests=pt, abs_ties=at,
                                    tau=tau, seed=s, M=M, SOL=SOL,
                                    NT=NTS[metric], tag=tag)


def _list(s, cast=int):
    return [cast(x) for x in str(s).split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="tests",
                    help="similarity: tests, init_tests, crosshair, codebleu (comma list)")
    ap.add_argument("--copies", default="1", help="number of adversary copies (comma list)")
    ap.add_argument("--ks",     default="2,3,6,7", help="the plurality parameter tau")
    ap.add_argument("--pt",     default="1", help="1 = rank by tests passed then size, 0 = size")
    ap.add_argument("--ties",   default="1", help="1 = abstain on ties, 0 = random tie-break")
    ap.add_argument("--seeds",  default="0")
    ap.add_argument("--tag",    default="")
    a = ap.parse_args()

    sweep(metrics=[m.strip() for m in a.metrics.split(",") if m.strip()],
          copies=_list(a.copies), ks=_list(a.ks), seeds=_list(a.seeds),
          pts=[bool(x) for x in _list(a.pt)], ats=[bool(x) for x in _list(a.ties)],
          tag=a.tag)
