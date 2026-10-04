"""Sanity check of the generated tests: run every test in data/reasoned_tests.json
against the canonical HumanEval solution of its task. A correct test passes on it.

Run inside the EvalPlus image (sandboxed, no network), from HumanEval/:
    docker run --rm --platform linux/amd64 --network none --memory 4g --pids-limit 512 \
      -v "$(pwd):/app" -w /app ganler/evalplus:latest python generation/validate_gen_tests.py

Reads  data/canonical.json    {task_id: prompt + canonical_solution}, from HumanEval
       data/reasoned_tests.json
Writes data/reasoned_test_validation.json   per task: share of tests that pass, failing tests
"""
import json, os, sys, tempfile, subprocess

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

RUNNER = r'''
import json, sys, signal, io, contextlib
def h(s,f): raise TimeoutError()
signal.signal(signal.SIGALRM, h)
cands, tests = json.load(open(sys.argv[1]))
out = []
for code in cands:
    ns, row = {}, []
    try:
        signal.alarm(5)
        with contextlib.redirect_stdout(io.StringIO()): exec(code, ns)
        signal.alarm(0)
    except BaseException:
        signal.alarm(0); out.append([0]*len(tests)); continue
    for t in tests:
        try:
            signal.alarm(1)
            with contextlib.redirect_stdout(io.StringIO()): exec(t, ns)
            signal.alarm(0); row.append(1)
        except BaseException:
            signal.alarm(0); row.append(0)
    out.append(row)
json.dump(out, open(sys.argv[2], "w"))
'''

canonical = json.load(open(os.path.join(DATA, "canonical.json")))
tests     = json.load(open(os.path.join(DATA, "reasoned_tests.json")))

report = {}
for tid, tsts in tests.items():
    if tid not in canonical or not tsts:
        continue
    with tempfile.TemporaryDirectory() as d:
        rp, ip, op = (os.path.join(d, f) for f in ("r.py","i.json","o.json"))
        open(rp,"w").write(RUNNER)
        json.dump([[canonical[tid]], tsts], open(ip,"w"))
        try:
            subprocess.run([sys.executable, rp, ip, op], timeout=120, capture_output=True)
            row = json.load(open(op))[0]
        except Exception:
            row = [0]*len(tsts)
    report[tid] = {"n_tests": len(tsts), "n_correct": sum(row),
                   "accuracy": sum(row)/len(tsts),
                   "wrong_tests": [t for t, ok in zip(tsts, row) if not ok]}

json.dump(report, open(os.path.join(DATA, "reasoned_test_validation.json"), "w"), indent=1)
accs = [r["accuracy"] for r in report.values()]
tot = sum(r["n_tests"] for r in report.values()); cor = sum(r["n_correct"] for r in report.values())
print(f"overall: {cor}/{tot} = {cor/tot:.1%}")
print(f"per-task: min {min(accs):.0%}, median {sorted(accs)[len(accs)//2]:.0%}")
for tid, r in sorted(report.items(), key=lambda kv: kv[1]["accuracy"])[:10]:
    print(f"  {tid}: {r['n_correct']}/{r['n_tests']} ({r['accuracy']:.0%})")