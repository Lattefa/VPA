"""Self-consistency, step 1: run the K sampled completions of one model against data/tests.json.

Usage:
    python test-sc-models.py --model z-ai_glm-4.5-air

Reads  data/SC/HE_sc5_{model}_completions.json   (from he_generation_exp.py)
Writes data/json_files/{model}_matrix.json        pass/fail per candidate and test
"""
import json, subprocess, tempfile, os, sys, argparse
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="slug as in data/SC/HE_sc5_{model}_completions.json")
model = ap.parse_args().model

COMPLETIONS = os.path.join(BASE, "data", "SC", f"HE_sc5_{model}_completions.json")
TESTS       = os.path.join(BASE, "data", "tests.json")
OUT         = os.path.join(BASE, "data", "json_files", f"{model}_matrix.json")
K           = 5                                            # candidates per task


tests = json.load(open(TESTS))
data  = json.load(open(COMPLETIONS))

by_task = {}
for tid, r in data.items():
    c = list(r.get("completions") or r.get("raw_responses") or [])
    by_task[tid] = (c + [""] * K)[:K]          

task_ids = sorted(by_task, key=lambda t: int(t.split("/")[1]))
print(f"{len(task_ids)} tasks, {K} candidates")

# report gaps
n_empty = sum(1 for t in task_ids for c in by_task[t] if not c.strip())
print(f"empty/missing cells: {n_empty} / {len(task_ids)*K}")
for i in range(K):
    c = sum(1 for t in task_ids if not by_task[t][i].strip())
    print(f"  candidate {i}: {c} empty")
no_tests = [t for t in task_ids if not tests.get(t)]
print(f"tasks with no tests: {len(no_tests)}")


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
    codes = by_task[tid]                                  
    if not ts:
        return tid, [[] for _ in codes]                  
    with tempfile.TemporaryDirectory() as d:
        rp, ip, op = (os.path.join(d, f) for f in ("r.py", "i.json", "o.json"))
        open(rp, "w").write(RUNNER)
        json.dump([codes, ts], open(ip, "w"))
        try:
            subprocess.run([sys.executable, rp, ip, op],
                           timeout=180, capture_output=True)
            rows = json.load(open(op))
        except Exception as e:
            print(f"  {tid} failed: {e}")
            rows = [[0]*len(ts) for _ in codes]            
    return tid, rows                                       

with ThreadPoolExecutor(8) as ex:                        
    matrix = dict(ex.map(run_task, task_ids))

json.dump({"k": K, "matrix": matrix,                     
           "n_tests": {t: len(tests.get(t, [])) for t in task_ids}},
          open(OUT, "w"))
print("wrote", OUT)