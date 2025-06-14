name2bench = {
    "bird-bench.yaml": "[Bird-Bench](https://bird-bench.github.io/)",
    "cy-bench.yaml": "[Cy-Bench](https://cybench.github.io/)",
    "gaia.yaml": "[GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard)" ,
    "kernel-bench.yaml": "[Kernel-Bench](https://scalingintelligence.stanford.edu/blogs/kernelbench/)",
    "osworld.yaml": "[OSWorld](https://os-world.github.io/)",
    "mle-bench.yaml": "[MLE-Bench](https://github.com/openai/mle-bench)",
    "swe-bench.yaml": "[SWE-Bench-Verified](https://openai.com/index/introducing-swe-bench-verified/)",
    "swe-lancer.yaml": "[SWE-Bench-Lancer](https://github.com/openai/SWELancer-Benchmark/)",
    "tau-bench.yaml": "[$\\\\tau$-Bench](https://sierra.ai/blog/benchmarking-ai-agents/)",
    "webarena.yaml": "[WebArena](https://webarena.dev/)"
}

import yaml
import csv

results = []
for name in name2bench:
    with open(name, "r") as f:
        data = yaml.safe_load(f)
    outcome_val = []
    challenge_val = []
    reporting = []
    for i in data:
        if i.startswith("I."):
            outcome_val.append(data[i]["score"])
        elif i.startswith("II."):
            challenge_val.append(data[i]["score"])
        elif i.startswith("III."):
            reporting.append(data[i]["score"])
        elif i in ["paper", "code"]:
            continue
        else:
            raise ValueError(f"Unknown benchmark type: {i}")
    outcome_val_score = sum(outcome_val)/len(outcome_val)
    challenge_val_score = sum(challenge_val)/len(challenge_val)
    reporting_score = sum(reporting)/len(reporting)
    avg_score = (outcome_val_score + challenge_val_score + reporting_score) / 3
    results.append([
        name2bench[name],
        f"{outcome_val_score*100:.1f}",
        f"{challenge_val_score*100:.1f}",
        f"{reporting_score*100:.1f}",
        f"{avg_score*100:.1f}",
    ])
    
# sort by average score
results.sort(key=lambda x: float(x[4]), reverse=True)
# write to csv
with open("../docs/assets/scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Benchmark", "Outcome Validity", "Task Validity", "Benchmark Reporting", "Overall"])
    for row in results:
        writer.writerow(row)
