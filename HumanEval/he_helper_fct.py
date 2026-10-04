
import time 
import re

def extract_code(text):
    """
    Extracts Python code from a model's response.
    """
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def ask(model, client,REASONING_MODELS,prompt, temperature=0.7, retries=3):
    """
    Queries the router_client. Note that temperature is increased 
    so the model generates diverse candidates across K samples.
    """
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, # Must be > 0 for diverse K sampling
            )
            
            if any(tag.lower() in model.lower() for tag in REASONING_MODELS):
                kwargs["max_tokens"] = 2048
                kwargs["reasoning_effort"] = "low"
            else:
                kwargs["max_tokens"] = 2048 

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
            
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 20 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return None


def merge_completions(data_dir="data"):
    """
    Merges data/completions/{model}.jsonl into data/all_completions.jsonl 
    and their EvalPlus results ({model}_eval_results.json) into
    data/humaneval_{base,plus}_results.csv (1 = passes, one column per model).
    """
    import os, json, glob, csv
    src = os.path.join(data_dir, "completions")
    models = sorted(os.path.basename(p)[:-len(".jsonl")]
                    for p in glob.glob(os.path.join(src, "*.jsonl")))

    with open(os.path.join(data_dir, "all_completions.jsonl"), "w", encoding="utf-8") as out:
        for m in models:
            with open(os.path.join(src, f"{m}.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        out.write(json.dumps({"task_id": e["task_id"], "model": m,
                                              "solution": e["solution"]}) + "\n")

    evals = {m: json.load(open(os.path.join(src, f"{m}_eval_results.json"),
                               encoding="utf-8"))["eval"] for m in models}
    tasks = sorted(set().union(*evals.values()), key=lambda t: int(t.split("/")[-1]))
    for status in ("base", "plus"):
        with open(os.path.join(data_dir, f"humaneval_{status}_results.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id"] + models)
            for t in tasks:
                w.writerow([t] + [int(evals[m][t][0][f"{status}_status"] == "pass")
                                  for m in models])
    print(f"merged {len(models)} models: {', '.join(models)}")
