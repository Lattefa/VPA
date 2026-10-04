import json, os, re
from collections import defaultdict
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from codebleu import calc_codebleu
except ImportError:
    raise ImportError("Please install the codebleu package: pip install codebleu")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
INPUT_FILE  = os.path.join(BASE, "data", "all_completions.jsonl")
OUTPUT_FILE = os.path.join(BASE, "data", "sim_codebleu_pairwise.json")
WEIGHTS     = (0.0, 0.0, 0.5, 0.5)


def clean_markdown(code_str: str) -> str:
    if not isinstance(code_str, str):
        return code_str
    m = re.search(r"```(?:python)?\s*(.*?)```", code_str, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else code_str.strip()


def _directed(ref: str, pred: str) -> float:
    try:
        return calc_codebleu([ref], [pred], weights=WEIGHTS, lang="python")["codebleu"]
    except Exception:
        return 0.0


def compute_pair(task_id, a_name, a_code, b_name, b_code) -> dict:
    """Symmetric CodeBLEU: mean of both directions."""
    if not a_code or not b_code:
        score = 0.0
    else:
        score = (_directed(a_code, b_code) + _directed(b_code, a_code)) / 2.0
    return {"task_id": task_id, "a": a_name, "b": b_name, "score": score}


def load_json_data(filepath: str) -> list:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


if __name__ == "__main__":
    print(f"Loading data from {INPUT_FILE}...")
    data = load_json_data(INPUT_FILE)

    tasks = defaultdict(list)
    for entry in data:
        tid = entry.get("task_id")
        if tid:
            tasks[tid].append(entry)

    print("Cleaning code and preparing jobs...")
    task_code, jobs = {}, []
    for tid, entries in tasks.items():
        if len(entries) < 2:
            continue
        code = {}
        for i, e in enumerate(entries):
            name = (e.get("model") or f"Unknown_Model_{i}").strip()
            code[name] = clean_markdown(e.get("solution", ""))
        task_code[tid] = code
        for a, b in combinations(sorted(code), 2):
            jobs.append((tid, a, code[a], b, code[b]))

    total = len(jobs)
    workers = max(1, os.cpu_count() - 1)
    print(f"Total pairs to compute: {total}")
    print(f"Starting ProcessPoolExecutor with {workers} workers...")

    pairwise = {t: {m: {} for m in c} for t, c in task_code.items()}

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(compute_pair, *j): (j[0], j[1], j[3]) for j in jobs}
        for fut in as_completed(futures):
            tid, a, b = futures[fut]
            done += 1
            try:
                r = fut.result()
                s = r["score"]
            except Exception as exc:
                print(f"[{done}/{total}] {tid} | {a} vs {b} -> Exception: {exc}")
                s = 0.0
            pairwise[tid][a][b] = s
            pairwise[tid][b][a] = s          # symmetry enforced here
            if done % 200 == 0 or done == total:
                print(f"[{done}/{total}] {tid} | {a} vs {b} -> {s:.4f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairwise, f, indent=2)

    print(f"\nWrote pairwise matrices for {len(pairwise)} tasks -> {OUTPUT_FILE}")