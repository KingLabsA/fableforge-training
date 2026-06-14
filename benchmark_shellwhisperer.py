#!/usr/bin/env python3
"""Benchmark ShellWhisperer-1.5B vs base Qwen2.5-Coder-1.5B-Instruct.

Tests accuracy, relevance, and conciseness of shell command generation
across 10 categories with 40+ test prompts.
"""

import torch
import os
import json
import time
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_PATH = "/tmp/fableforge/models/shellwhisperer-1.5b-lora"
SYSTEM_PROMPT = "You are ShellWhisperer, an expert in command-line tools, shell scripting, and system administration. Provide concise, accurate commands with brief explanations."

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ════════════════════════════════════════════════════════
# Test categories and prompts
# ════════════════════════════════════════════════════════

TESTS = {
    "File Operations": [
        "Find all Python files modified in the last week",
        "Recursively delete all __pycache__ directories",
        "Count the total number of lines in all .py files in a directory",
        "Find and replace 'TODO' with 'FIXME' across all Python files",
    ],
    "Process Management": [
        "Kill a process listening on port 3000",
        "Show the top 5 memory-consuming processes",
        "List all zombie processes",
        "Monitor a log file in real-time filtering for ERROR",
    ],
    "Networking": [
        "Show all listening TCP ports",
        "Test if port 443 is reachable on example.com",
        "Show DNS resolution for a domain",
        "Trace the network path to google.com",
    ],
    "Text Processing": [
        "Sort a CSV file by the 3rd column numerically",
        "Remove duplicate lines from a file keeping order",
        "Extract the 2nd column from a tab-delimited file",
        "Count occurrences of each unique line in a file",
    ],
    "Git": [
        "Show the last commit that modified a specific file",
        "Undo the last commit keeping changes staged",
        "List all branches merged into main",
        "Delete all local branches already merged into main",
    ],
    "Docker": [
        "Remove all stopped Docker containers",
        "Show logs from a container from the last hour",
        "Run a container with a 512MB memory limit",
        "Copy a file from a running container to the host",
    ],
    "Python": [
        "Start a simple HTTP server on port 8000",
        "Pretty-print a JSON file from the command line",
        "Profile a Python script to find bottlenecks",
        "Sort imports in a Python project with isort",
    ],
    "Security": [
        "Find all SUID-root binaries on the system",
        "Generate a strong random password",
        "Check SSL certificate expiration for a domain",
        "Find world-writable files in /etc",
    ],
    "System Admin": [
        "Show disk usage sorted by size for /home",
        "Show failed systemd services",
        "Check how long the system has been running",
        "Restart a systemd service and follow its logs",
    ],
    "Advanced Shell": [
        "Create a temporary directory and clean up on exit",
        "Run a command every 5 seconds",
        "Replace newlines with spaces in a file",
        "Batch rename .jpeg files to .jpg",
    ],
}

# ════════════════════════════════════════════════════════
# Scoring rubric (keyword-based)
# ════════════════════════════════════════════════════════

CORRECT_COMMANDS = {
    "File Operations": [
        ["find", "-mtime", "-7", "*.py"],
        ["find", "-name", "__pycache__", "-delete", "-type", "d"],
        ["find", "-name", "*.py", "wc", "-l"],
        ["find", "-name", "*.py", "sed", "TODO", "FIXME"],
    ],
    "Process Management": [
        ["lsof", ":3000", "kill", "-9"],
        ["ps", "--sort=-%mem", "head"],
        ["ps", "awk", "Z"],
        ["tail", "-f", "grep", "ERROR"],
    ],
    "Networking": [
        ["ss", "-tlnp", "lsof", "-i", "netstat"],
        ["nc", "-zv", "curl"],
        ["dig", "nslookup"],
        ["mtr", "traceroute", "tracepath"],
    ],
    "Text Processing": [
        ["sort", "-k3", "-n", "-t"],
        ["awk", "!seen", "uniq", "sort", "-u"],
        ["cut", "-f2", "-d"],
        ["sort", "uniq", "-c"],
    ],
    "Git": [
        ["git", "log", "-1", "--follow"],
        ["git", "reset", "--soft", "HEAD~1"],
        ["git", "branch", "--merged"],
        ["git", "branch", "--merged", "grep", "-v"],
    ],
    "Docker": [
        ["docker", "container", "prune"],
        ["docker", "logs", "--since"],
        ["docker", "run", "-m", "--memory"],
        ["docker", "cp"],
    ],
    "Python": [
        ["python", "-m", "http.server", "8000"],
        ["python", "-m", "json.tool"],
        ["python", "-m", "cProfile"],
        ["python", "-m", "isort"],
    ],
    "Security": [
        ["find", "-perm", "-4000", "SUID"],
        ["openssl", "rand", "-base64"],
        ["openssl", "s_client", "x509", "-dates"],
        ["find", "/etc", "-perm", "-o+w"],
    ],
    "System Admin": [
        ["du", "-sh", "sort", "-rh"],
        ["systemctl", "--failed"],
        ["uptime"],
        ["systemctl", "restart", "journalctl", "-f"],
    ],
    "Advanced Shell": [
        ["mktemp", "trap", "rm", "-rf"],
        ["watch", "-n", "5"],
        ["tr", "'\\n'", "' '"],
        ["for", "mv", ".jpeg", ".jpg"],
    ],
}

def score_response(response, category, index):
    score = 0
    details = ""
    resp_lower = response.lower()
    
    keywords = CORRECT_COMMANDS.get(category, [[]])[min(index, len(CORRECT_COMMANDS.get(category, [[]])) - 1)]
    matched = [kw for kw in keywords if kw.lower() in resp_lower]
    
    if len(matched) >= 2:
        score = 2  # Good: contains key correct elements
        details = f"Matched: {matched}"
    elif len(matched) == 1:
        score = 1  # Partial: at least one correct element
        details = f"Partial: {matched}"
    else:
        score = 0  # Missed
        details = "No key commands matched"
    
    # Bonus: contains actual command syntax (backticks or command-like patterns)
    if "`" in response or "$" in response or " | " in response:
        score = min(score + 0.5, 2.5)
        details += " +command_syntax"
    
    # Penalty: way too verbose (>500 chars for a simple command)
    if len(response) > 500 and category != "Advanced Shell":
        score = max(score - 0.3, 0)
        details += " -verbose"
    
    return score, details

def generate(model, tokenizer, prompt, max_tokens=150):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        )
    elapsed = time.time() - start
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip(), elapsed

# ════════════════════════════════════════════════════════
# Load models
# ════════════════════════════════════════════════════════

print("=" * 70)
print("ShellWhisperer-1.5B Benchmark")
print("=" * 70)

print(f"\nDevice: {device}")
print(f"Base model: {BASE_MODEL}")
print(f"LoRA path: {LORA_PATH}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(device)
base_model.eval()

# Load fine-tuned model
print("Loading fine-tuned model (base + LoRA)...")
ft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
ft_model.eval()

# ════════════════════════════════════════════════════════
# Run benchmark
# ════════════════════════════════════════════════════════

results = {
    "base": {"total_score": 0, "total_time": 0, "by_category": {}},
    "finetuned": {"total_score": 0, "total_time": 0, "by_category": {}},
}

total_prompts = sum(len(v) for v in TESTS.values())
current = 0

print(f"\nRunning benchmark: {total_prompts} prompts x 2 models = {total_prompts * 2} generations")
print("-" * 70)

for category, prompts in TESTS.items():
    for idx, prompt in enumerate(prompts):
        current += 1
        print(f"\n[{current}/{total_prompts}] {category}: {prompt[:60]}...")
        
        # Base model
        base_resp, base_time = generate(base_model, tokenizer, prompt)
        base_score, base_detail = score_response(base_resp, category, idx)
        results["base"]["total_score"] += base_score
        results["base"]["total_time"] += base_time
        
        # Fine-tuned model (rebase for each test to avoid state issues)
        ft_resp, ft_time = generate(ft_model, tokenizer, prompt)
        ft_score, ft_detail = score_response(ft_resp, category, idx)
        results["finetuned"]["total_score"] += ft_score
        results["finetuned"]["total_time"] += ft_time
        
        # Track per-category
        if category not in results["base"]["by_category"]:
            results["base"]["by_category"][category] = {"score": 0, "count": 0}
            results["finetuned"]["by_category"][category] = {"score": 0, "count": 0}
        results["base"]["by_category"][category]["score"] += base_score
        results["base"]["by_category"][category]["count"] += 1
        results["finetuned"]["by_category"][category]["score"] += ft_score
        results["finetuned"]["by_category"][category]["count"] += 1
        
        # Print comparison
        winner = "FT" if ft_score > base_score else ("BASE" if base_score > ft_score else "TIE")
        print(f"  BASE score={base_score:.1f} ({base_time:.1f}s) | FT score={ft_score:.1f} ({ft_time:.1f}s) | {winner}")
        print(f"  BASE: {base_resp[:120]}...")
        print(f"  FT:   {ft_resp[:120]}...")

# ════════════════════════════════════════════════════════
# Results
# ════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

base_avg = results["base"]["total_score"] / total_prompts
ft_avg = results["finetuned"]["total_score"] / total_prompts
base_time_avg = results["base"]["total_time"] / total_prompts
ft_time_avg = results["finetuned"]["total_time"] / total_prompts

print(f"\n{'Metric':<30} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
print("-" * 65)
print(f"{'Total Score (max {:.0f})'.format(total_prompts * 2.5):<30} {results['base']['total_score']:>10.1f} {results['finetuned']['total_score']:>12.1f} {results['finetuned']['total_score'] - results['base']['total_score']:>+10.1f}")
print(f"{'Avg Score per prompt':<30} {base_avg:>10.2f} {ft_avg:>12.2f} {ft_avg - base_avg:>+10.2f}")
print(f"{'Total Time (s)':<30} {results['base']['total_time']:>10.1f} {results['finetuned']['total_time']:>12.1f} {results['finetuned']['total_time'] - results['base']['total_time']:>+10.1f}")
print(f"{'Avg Time per prompt (s)':<30} {base_time_avg:>10.2f} {ft_time_avg:>12.2f} {ft_time_avg - base_time_avg:>+10.2f}")
print(f"{'Prompts tested':<30} {total_prompts:>10} {total_prompts:>12}")

print(f"\n{'Category':<25} {'Base Avg':>10} {'FT Avg':>10} {'Improvement':>12}")
print("-" * 60)
for cat in TESTS:
    b = results["base"]["by_category"][cat]
    f = results["finetuned"]["by_category"][cat]
    b_avg = b["score"] / b["count"]
    f_avg = f["score"] / f["count"]
    delta = f_avg - b_avg
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(f"{cat:<25} {b_avg:>10.2f} {f_avg:>10.2f} {delta:>+10.2f} {arrow}")

# Win rate
base_wins = sum(1 for cat in TESTS for i, _ in enumerate(TESTS[cat]) 
                if score_response(
                    generate(base_model, tokenizer, _)[0], cat, i)[0] > 
                    score_response(
                        generate(ft_model, tokenizer, _)[0], cat, i)[0])

improvement_pct = ((ft_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0
print(f"\nFine-tuned improvement: {improvement_pct:+.1f}%")
print(f"Fine-tuned model {'OUTPERFORMS' if ft_avg > base_avg else 'UNDERPERFORMS'} base model")

# Save results
output_path = "/tmp/fableforge/models/benchmark_results.json"
with open(output_path, "w") as f:
    json.dump({
        "base_score": results["base"]["total_score"],
        "finetuned_score": results["finetuned"]["total_score"],
        "base_avg": base_avg,
        "finetuned_avg": ft_avg,
        "improvement_pct": improvement_pct,
        "total_prompts": total_prompts,
        "by_category": {
            cat: {
                "base_avg": results["base"]["by_category"][cat]["score"] / results["base"]["by_category"][cat]["count"],
                "finetuned_avg": results["finetuned"]["by_category"][cat]["score"] / results["finetuned"]["by_category"][cat]["count"],
            }
            for cat in TESTS
        }
    }, f, indent=2)
print(f"\nResults saved to {output_path}")