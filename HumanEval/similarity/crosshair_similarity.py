
import ast
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed


STATUS_CODE = {
    "error": -1,
    "divergence_found": 0,
    "equivalent": 1,
    "timeout": 2,
}


def get_function_name(code_str: str) -> str:
    """Parses the AST to find the name of the function defined in the code."""
    try:
        tree = ast.parse(code_str)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        return None
    return None


def run_crosshair_diff(code1: str, code2: str, timeout: int = 30) -> dict:
    """Runs crosshair diffbehavior on two code snippets to find counterexamples."""
    func_name = get_function_name(code1)

    if not func_name:
        return {"status": "error", "message": "Could not parse function name (Syntax Error in Code 1)."}

    if get_function_name(code2) != func_name:
        return {"status": "error", "message": "Function names do not match or Syntax Error in Code 2."}

    with tempfile.TemporaryDirectory() as tempdir:
        mod1_path = os.path.join(tempdir, "mod1.py")
        mod2_path = os.path.join(tempdir, "mod2.py")

        with open(mod1_path, "w", encoding="utf-8") as f1, open(mod2_path, "w", encoding="utf-8") as f2:
            f1.write(code1)
            f2.write(code2)

        env = os.environ.copy()
        env["PYTHONPATH"] = tempdir + os.pathsep + env.get("PYTHONPATH", "")

        try:
            cmd = ["crosshair", "diffbehavior", f"mod1.{func_name}", f"mod2.{func_name}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )

            output = result.stdout.strip() or result.stderr.strip()

            if "differs" in output.lower() or "when called with" in output.lower():
                return {"status": "divergence_found", "counterexample": output}
            elif result.returncode == 0:
                return {"status": "equivalent", "message": "No counterexamples found (within timeout/bounds)."}
            else:
                return {"status": "error", "message": f"CrossHair error: {output}"}

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": f"CrossHair timed out after {timeout} seconds."}


def load_json_data(filepath: str) -> list:
    """Robustly loads either a JSON array or JSON Lines format."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
    input_file = os.path.join(base, "data", "all_completions.jsonl")
    output_file = os.path.join(base, "data", "crosshair_similarity_results.json")

    print(f"Loading data from {input_file}...")
    data = load_json_data(input_file)

    tasks = defaultdict(list)
    for entry in data:
        task_id = entry.get('task_id')
        if task_id:
            tasks[task_id].append(entry)

    task_matrices = {}
    jobs = []

    print("Preparing jobs for parallel execution...")
    for task_id, entries in tasks.items():
        if len(entries) < 2:
            continue

        model_to_code = {
            entry.get('model', f'Unknown_Model_{i}'): entry.get('solution', '')
            for i, entry in enumerate(entries)
        }
        models = list(model_to_code.keys())

        # Pre-compute which models have broken/unparseable code
        broken = {m for m, code in model_to_code.items() if get_function_name(code) is None}

        # Initialize NxN matrix: diagonal 1 (or -1 if the model's own code is broken)
        task_matrices[task_id] = {
            m1: {m2: (-1 if (m1 in broken or m2 in broken)
                      else (1 if m1 == m2 else 0))
                 for m2 in models}
            for m1 in models
        }

        # Only schedule comparisons between parseable candidates;
        # broken ones are already marked -1 across their row/column
        pairs = [(m1, m2) for m1, m2 in combinations(models, 2)
                 if m1 not in broken and m2 not in broken]
        for m1, m2 in pairs:
            jobs.append({
                'task_id': task_id,
                'm1': m1,
                'm2': m2,
                'code1': model_to_code[m1],
                'code2': model_to_code[m2]
            })

    total_jobs = len(jobs)
    print(f"Total comparisons to run: {total_jobs}")

    optimal_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Starting ProcessPoolExecutor with {optimal_workers} workers...")

    completed = 0
    with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        future_to_job = {
            executor.submit(run_crosshair_diff, job['code1'], job['code2']): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            task_id, m1, m2 = job['task_id'], job['m1'], job['m2']
            completed += 1

            try:
                diff_result = future.result()
                code = STATUS_CODE[diff_result['status']]
            except Exception as exc:
                diff_result = {"status": "error", "message": str(exc)}
                code = -1

            # Populate matrix symmetrically
            task_matrices[task_id][m1][m2] = code
            task_matrices[task_id][m2][m1] = code

            status_msg = {
                0: "Counterexample found!",
                1: "Equivalent.",
                2: "Timeout.",
                -1: f"Error: {diff_result.get('message', 'Unknown')}",
            }[code]
            print(f"[{completed}/{total_jobs}] {task_id} | {m1} vs {m2} -> {status_msg}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(task_matrices, f, indent=4)

    print(f"\nAnalysis complete. Evaluated {len(task_matrices)} HumanEval tasks.")
    print(f"Similarity matrices saved to {output_file}")