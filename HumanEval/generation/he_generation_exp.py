import argparse
import os
import sys
import json
import time
import re
from groq import Groq
from datasets import load_dataset
from openai import OpenAI
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # HumanEval/
sys.path.insert(0, BASE)
from he_helper_fct import extract_code, ask


def main(model,K,dataset,N_QUESTIONS):
    
    
    router_client = OpenAI(base_url="https://openrouter.ai/api/v1",
            api_key="Your API key here")


    
    REASONING_MODELS = ("gpt-oss", "DeepSeek", "gemma", "laguna", "nemotron")
    slug = model.replace("/", "_").replace(":", "_")

    OUTPUT_JSON_PATH = os.path.join(BASE, "data", "SC", f"HE_sc{K}_{slug}_completions.json")
    print(OUTPUT_JSON_PATH)

    print("Loading HumanEval dataset...")
    raw_dataset = load_dataset("openai/openai_humaneval", split="test")
    subset = raw_dataset.select(range(min(N_QUESTIONS, len(raw_dataset))))

    # 2. Load existing JSON progress (if any)
    results = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        print(f"Reading existing results from {OUTPUT_JSON_PATH}...")
        with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print("Existing JSON is empty or invalid. Starting fresh.")

    print("\nStarting HumanEval generation loop...")

    for item in subset:
        task_id = item["task_id"]       
        prompt_text = item["prompt"]    
        
        # Check if we already have K completions for this task
        existing_data = results.get(task_id, {})
        existing_completions = existing_data.get("completions", [])
        
        if len(existing_completions) >= K:
            continue
            
        print(f"\nFetching {K} samples for {task_id} using {model}...")
        
        full_prompt = (
            "Complete the following Python function. "
            "Reply ONLY with the raw Python code covering the imports, function signature, and the complete body. "
            "Do not include any explanations, markdown formatting, or triple backticks.\n\n"
            f"{prompt_text}"
        )
        
        # Load existing ones if the script crashed midway through generating K samples for this task
        completions = existing_completions.copy()
        raw_responses = existing_data.get("raw_responses", []).copy()
        
        # Generate the remaining samples up to K
        for i in range(len(completions), K):
            print(f"  -> Generating sample {i+1}/{K}...")
            #model, client,REASONING_MODELS,prompt
            raw_answer_text = ask(model,router_client,REASONING_MODELS, full_prompt, temperature=0.7)
            clean_code = extract_code(raw_answer_text or "")
            
            completions.append(clean_code)
            raw_responses.append(raw_answer_text)
            time.sleep(2.5)
            
        # Update dictionary
        results[task_id] = {
            "task_id": task_id,
            "model": model,
            "completions": completions,
            "raw_responses": raw_responses
        }
        

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        print(f"Saved {task_id}")

    print("\nGeneration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample completions for HumanEval tasks"
                    "-> data/SC/HE_sc{K}_{model}_completions.json")
    parser.add_argument('-m', '--model', type=str, required=True, help='OpenRouter model id, e.g. z-ai/glm-4.5-air')
    parser.add_argument('-K', '--sc', type=int, default=5, help='completions sampled per task')
    parser.add_argument('-d', '--dataset', default='HumanEval')
    parser.add_argument('-n', '--n_qst', type=int, default=164, help='number of tasks')
    args = parser.parse_args()

    main(args.model, args.sc, args.dataset, args.n_qst)
