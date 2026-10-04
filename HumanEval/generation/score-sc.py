"""Self-consistency, step 2: pick one of the K candidates per task and write the model's completion file.

Pick = candidates grouped by identical pass/fail fingerprint on data/tests.json,
best group by size x tests passed, lowest index on ties.

Usage:
    python score-sc.py --model z-ai_glm-4.5-air --name glm-4.5-air

Reads  data/SC/HE_sc5_{model}_completions.json, data/json_files/{model}_matrix.json
Writes data/json_files/{model}_picks.json, data/json_files/sc-diag-{model}.json
       data/completions/{name}.jsonl                 (merged by he_helper_fct.merge_completions)
"""
import json, os, argparse
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="slug as in data/SC/HE_sc5_{model}_completions.json")
ap.add_argument("--name",  default=None, help="model name in data/completions/ (default: --model)")
args  = ap.parse_args()
model = args.model
name  = args.name or model

DATA = os.path.join(BASE, "data")
full_completion_file   = os.path.join(DATA, "SC", f"HE_sc5_{model}_completions.json")
picks_file             = os.path.join(DATA, "json_files", f"{model}_picks.json")
completion_output_file = os.path.join(DATA, "completions", f"{name}.jsonl")

d = json.load(open(os.path.join(DATA, "json_files", f"{model}_matrix.json")))
K, matrix = d["k"], d["matrix"]                           

picks = {}
diag  = {}

for tid, rows in matrix.items():                          
    live = {i: fp for i, fp in enumerate(rows) if fp}      
    if not live:
        picks[tid] = 0
        diag[tid] = {"reason": "no_tests", "groups": 0}
        continue

    # group candidates by identical pass/fail fingerprint
    groups = defaultdict(list)
    for i, fp in live.items():
        groups[tuple(fp)].append(i)

    scored = [(len(ms) * sum(fp), len(ms), sum(fp), sorted(ms))
              for fp, ms in groups.items()]
    best_score, gsize, npass, winners = max(scored)
    n_tests = len(next(iter(live.values())))

    if best_score == 0:                       # nothing passed any test
        nonempty = [i for i in range(len(rows)) if rows[i]]
        picks[tid] = nonempty[0] if nonempty else 0
        diag[tid] = {"reason": "all_zero", "groups": len(groups),
                     "n_tests": n_tests}
    else:
        picks[tid] = winners[0]               # deterministic tie-break
        diag[tid] = {"reason": "ok", "groups": len(groups),
                     "group_size": gsize, "tests_passed": npass,
                     "n_tests": n_tests,
                     "agreement": gsize / len(live),       
                     "solved": npass == n_tests,            
                     "pass_counts": [sum(fp) for fp in rows]}  

json.dump(picks, open(picks_file , "w"), indent=1)


ok = [v for v in diag.values() if v["reason"] == "ok"]

print("\n=== which candidate won ===")                    
for i, c in sorted(Counter(picks.values()).most_common()):
    print(f"  candidate {i}: {c:4d}")

print("\n=== agreement (winning cluster size) ===")       
cs = Counter(v["group_size"] for v in ok)
for k in sorted(cs):
    print(f"  {k}/{K} agreed: {cs[k]} tasks")

print("\n=== group structure ===")
gs = Counter(v.get("groups", 0) for v in diag.values())
for k in sorted(gs):
    print(f"  {k} group(s): {gs[k]} tasks")
print("  all-zero tasks:", sum(1 for v in diag.values()
                               if v["reason"] == "all_zero"))
print("  no-test tasks: ", sum(1 for v in diag.values()
                               if v["reason"] == "no_tests"))


scored_tasks = [v for v in diag.values() if v.get("n_tests")]
n = len(scored_tasks)
if n:
    pass1 = sum(sum(p == v["n_tests"] for p in v.get("pass_counts", [])) / K
                for v in scored_tasks) / n
    passk = sum(any(p == v["n_tests"] for p in v.get("pass_counts", []))
                for v in scored_tasks) / n
    sc    = sum(v.get("solved", False) for v in scored_tasks) / n
    print(f"\npass@1 (avg over {K}): {pass1:.3f}")
    print(f"pass@{K} (any):        {passk:.3f}")
    print(f"self-consistency pick: {sc:.3f}")

json.dump(diag, open(os.path.join(DATA, "json_files", f"sc-diag-{model}.json"), "w"), indent=1)

###### generate completion file from the picks #####


picks = json.load(open(picks_file))
data  = json.load(open(full_completion_file))               
K     = 5

rows = {}                                                 
for tid, r in data.items():
    c = list(r.get("completions") or r.get("raw_responses") or [])
    rows[tid] = (c + [""] * K)[:K]

task_ids = sorted(rows, key=lambda t: int(t.split("/")[1]))

def dump(name, chooser):
    n_empty = 0
    with open(name, "w") as f:
        for tid in task_ids:                               
            cands = rows[tid]
            i = chooser(tid, cands)
            code = cands[i] if 0 <= i < len(cands) else ""
            if not code.strip():
                n_empty += 1
            f.write(json.dumps({"task_id": tid, "solution": code}) + "\n")
    print(f"wrote {name}  ({len(task_ids)} tasks, {n_empty} empty)")

dump(completion_output_file, lambda t, c: picks.get(t, 0))      
