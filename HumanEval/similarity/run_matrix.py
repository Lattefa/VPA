"""Pass/fail matrix: every completion against every generated test of its task.

Run inside the EvalPlus image (sandboxed, no network):
    docker run --rm --platform linux/amd64 --network none --memory 4g --pids-limit 512 \
      -v "$(pwd):/app" -w /app ganler/evalplus:latest \
      python similarity/run_matrix.py --tests data/tests.json --out data/init_test_matrix.json
    (same with --tests data/reasoned_tests.json --out data/rbased_matrix.json)

Writes data/all_completions.jsonl, data/humaneval_{base,plus}_results.csv (merge_completions)
   and {"models": [...], "matrix": {task: {model: [0/1 per test]}}}
"""
import json, subprocess, tempfile, os, sys, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
sys.path.insert(0, BASE)
from he_helper_fct import merge_completions

ap = argparse.ArgumentParser()
ap.add_argument("--tests", default="data/tests.json")
ap.add_argument("--out",   default="data/init_test_matrix.json")
args = ap.parse_args()

COMPLETIONS = os.path.join(BASE, "data", "all_completions.jsonl")
TESTS       = os.path.join(BASE, args.tests)
OUT         = os.path.join(BASE, args.out)

# ---- merge the per-model completion files, then load, keyed by (task, model) ----
merge_completions(os.path.join(BASE, "data"))
tests = json.load(open(TESTS))#json.load(open(TESTS))
by_task = defaultdict(dict)
models = []
for line in open(COMPLETIONS):
    r = json.loads(line)
    m = r["model"]
    if m not in models:
        models.append(m)                       # fixed order = file order
    by_task[r["task_id"]][m] = r.get("solution") or r.get("completion") or ""

task_ids = sorted(by_task, key=lambda t: int(t.split("/")[1]))
print(f"{len(task_ids)} tasks, {len(models)} models")

# report gaps
missing = {t: [m for m in models if not by_task[t].get(m, "").strip()]
           for t in task_ids}
n_empty = sum(len(v) for v in missing.values())
print(f"empty/missing cells: {n_empty} / {len(task_ids)*len(models)}")
for m in models:
    c = sum(1 for t in task_ids if not by_task[t].get(m, "").strip())
    print(f"  {m}: {c} empty")


RUNNER = r'''
import json, sys, signal, io, contextlib
def h(s, f): raise TimeoutError()
signal.signal(signal.SIGALRM, h)
codes, tests = json.load(open(sys.argv[1]))
out = []
for code in codes:
    if not code.strip():
        out.append([0]*len(tests)); continue
    ns, row = {}, []
    try:
        signal.alarm(5)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, ns)
        signal.alarm(0)
    except BaseException:
        signal.alarm(0); out.append([0]*len(tests)); continue
    for t in tests:
        try:
            signal.alarm(1)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(t, ns)
            signal.alarm(0); row.append(1)
        except BaseException:
            signal.alarm(0); row.append(0)
    out.append(row)
json.dump(out, open(sys.argv[2], "w"))
'''

def run_task(tid):
    ts = tests.get(tid, [])
    codes = [by_task[tid].get(m, "") for m in models]      # fixed order
    if not ts:
        return tid, {m: [] for m in models}
    with tempfile.TemporaryDirectory() as d:
        rp, ip, op = (os.path.join(d, f) for f in ("r.py", "i.json", "o.json"))
        open(rp, "w").write(RUNNER)
        json.dump([codes, ts], open(ip, "w"))
        try:
            subprocess.run([sys.executable, rp, ip, op],
                           timeout=250, capture_output=True)
            rows = json.load(open(op))
        except Exception as e:
            print(f"  {tid} failed: {e}")
            rows = [[0]*len(ts) for _ in models]
    return tid, {m: rows[i] for i, m in enumerate(models)}

with ThreadPoolExecutor(16) as ex:
    matrix = dict(ex.map(run_task, task_ids))

json.dump({"models": models, "matrix": matrix}, open(OUT, "w"))
print("wrote", OUT)