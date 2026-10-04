import json, os
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
OUT  = os.path.join(BASE, "data", "adj"); os.makedirs(OUT, exist_ok=True)

FP_FILE = os.path.join(BASE, "data", "rbased_matrix.json")
IT_FILE = os.path.join(BASE, "data", "init_test_matrix.json")
CH_FILE = os.path.join(BASE, "data", "crosshair_similarity_results.json")
CB_FILE = os.path.join(BASE, "data", "sim_codebleu_pairwise.json")
COMPL   = os.path.join(BASE, "data", "all_completions.jsonl")


def task_key(t):
    tail = t.split("/")[-1]
    return (0, int(tail)) if tail.isdigit() else (1, t)


def _s(m):
    return (m or "").strip()


fp_doc = json.load(open(FP_FILE, encoding="utf-8"))
it_doc = json.load(open(IT_FILE, encoding="utf-8"))
MODELS = [_s(m) for m in fp_doc["models"]]

FP = {t: {_s(m): v for m, v in rows.items()}
      for t, rows in fp_doc["matrix"].items()}
IT = {t: {_s(m): v for m, v in rows.items()}
      for t, rows in it_doc["matrix"].items()}
CH = {t: {_s(x): {_s(y): v for y, v in row.items()} for x, row in rows.items()}
      for t, rows in json.load(open(CH_FILE, encoding="utf-8")).items()}
CB = {t: {_s(x): {_s(y): v for y, v in row.items()} for x, row in rows.items()}
      for t, rows in json.load(open(CB_FILE, encoding="utf-8")).items()}

if set(it_doc["models"]) and {_s(m) for m in it_doc["models"]} != set(MODELS):
    print("warn: init_test_matrix models differ from rbased_matrix models")

IDX, N = {m: i for i, m in enumerate(MODELS)}, len(MODELS)

TASKS = sorted(set(FP) & set(CH) & set(CB), key=task_key)
for nm, d in (("fp", FP), ("ch", CH), ("cb", CB)):
    if set(d) - set(TASKS):
        print(f"warn: {len(set(d) - set(TASKS))} tasks only in {nm}, dropped")

# init_tests lives on its own task set
TASKS_IT = sorted(set(IT), key=task_key)
print(f"init_tests: {len(TASKS_IT)} tasks "
      f"({len(set(TASKS_IT) - set(TASKS))} not in the shared set)")

SOL = defaultdict(dict)
with open(COMPL, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            e = json.loads(line)
            SOL[e["task_id"]][(e.get("model") or "").strip()] = \
                (e.get("solution") or "").strip()


def _dense(rows):
    a = np.full((N, N), np.nan)
    for x, row in (rows or {}).items():
        if x in IDX:
            for y, v in row.items():
                if y in IDX:
                    a[IDX[x], IDX[y]] = v
    return a


def sim_crosshair(tid):
    raw = _dense(CH.get(tid))
    return (raw == 1).astype(np.float32), ~np.isnan(raw), np.diag(raw == 1).copy()


def sim_codebleu(tid):
    raw   = _dense(CB.get(tid))
    known = ~np.isnan(raw)
    both  = known & known.T
    sym   = np.where(both, (np.nan_to_num(raw) + np.nan_to_num(raw.T)) / 2,
                     np.nan_to_num(raw) + np.nan_to_num(raw.T))
    np.fill_diagonal(sym, 1.0)
    return sym.astype(np.float32), known | known.T, np.ones(N, bool)


def _fingerprint_sim(src, tid, label):
    """Fraction of tests on which two models agree (pass AND fail)."""
    rows = [src.get(tid, {}).get(m) for m in MODELS]
    have = np.array([r is not None for r in rows])
    k    = max((len(r) for r in rows if r is not None), default=1)
    if any(r is not None and len(r) != k for r in rows):
        raise ValueError(f"{tid}: ragged {label} fingerprints")
    fp = np.array([r if r is not None else [0] * k for r in rows], dtype=np.int8)
    s  = (fp[:, None, :] == fp[None, :, :]).mean(axis=2).astype(np.float32)
    return s, have[:, None] & have[None, :], have


def sim_tests(tid):
    return _fingerprint_sim(FP, tid, "rbased")


def sim_init_tests(tid):
    return _fingerprint_sim(IT, tid, "init")


BUILDERS = {"crosshair":  sim_crosshair,
            "codebleu":   sim_codebleu,
            "tests":      sim_tests,
            "init_tests": sim_init_tests}


PASSED_SRC = {"init_tests": IT}
TASK_LIST  = {"init_tests": TASKS_IT}


def build(metric):
    fn    = BUILDERS[metric]
    psrc  = PASSED_SRC.get(metric, FP)
    tasks = TASK_LIST.get(metric, TASKS)

    S = np.zeros((len(tasks), N, N), np.float32)
    K = np.zeros((len(tasks), N, N), bool)
    V = np.zeros((len(tasks), N), bool)
    P = np.zeros((len(tasks), N), np.int16)          # tests passed, for tie-breaks

    for t, tid in enumerate(tasks):
        s, k, v = fn(tid)
        e = np.array([bool(SOL.get(tid, {}).get(m)) for m in MODELS])
        S[t], K[t], V[t] = s, k & e[:, None] & e[None, :], v & e
        P[t] = [sum(psrc.get(tid, {}).get(m) or []) for m in MODELS]

    np.savez_compressed(
        os.path.join(OUT, f"{metric}_sim.npz"),
        sim=S, known=K, valid=V, passed=P,
        tasks=np.array(tasks), models=np.array(MODELS))

    off  = ~np.eye(N, dtype=bool)
    elig = V[:, :, None] & V[:, None, :]
    print(f"\n{metric}  tasks={len(tasks)}")
    print(f"  ineligible : {(~elig[:, off]).mean():.3f}")
    print(f"  unscored   : {((~K) & elig)[:, off].mean():.3f}")
    print(f"  unknown={(~K[:, off]).mean():.3f} "
          f"valid/task={V.sum(1).mean():5.2f} "
          f"sim p50={np.median(S[:, off]):.3f} "
          f"frac==1={(S[:, off] == 1).mean():.3f} "
          f"mean passed={P[V].mean():.2f}")
    return S, K, V


if __name__ == "__main__":
    for m in BUILDERS:
        build(m)

