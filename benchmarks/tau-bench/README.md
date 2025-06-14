# Two Task Validity Issues in Tau-Bench

## Overview
First, Tau-Bench intentionally design challenges that cannot be resolved, taking 
up 38% of the airline partition and 6% of the retail partition. In these 
challenges, agents can pass evaluations as long as the database is unchanged. 
Therefore, a trivial agent that does nothing can succeed on these challenges. 
Second, 2% of airline challenges and 3.6% of retail challenges use texts that 
are directly from the database as the ground truth for the substring matching. 
On these challenges, a spamming agent that dumps everything in the database can 
pass evaluations.

To fix the issue, we can remove intentionally unsolvable challenges and 
challenges that use texts directly from the database for substring matching.

## Reproducing the Issues

### Issue 1: A trivial do-nothing agent scores high pass@k and pass^k
To reproduce the first issue, clone the Tau-Bench repository and apply `exploit-1.patch`:
```bash
git clone https://github.com/sierra-research/tau-bench && cd tau-bench
git checkout 14bf0ef52e595922d597a38f32d3e8c0dce3a8f8
git apply ../exploit-1.patch
```

Then, run the trivial do-nothing agent by
```bash
python run.py \
      --agent-strategy tool-calling \
      --env retail \
      --model gpt-4o \
      --model-provider openai \
      --user-model gpt-4o \
      --user-model-provider openai \
      --user-strategy llm \
      --max-concurrency 10
python run.py \
      --agent-strategy tool-calling \
      --env airline \
      --model gpt-4o \
      --model-provider openai \
      --user-model gpt-4o \
      --user-model-provider openai \
      --user-strategy llm \
      --max-concurrency 10
```

You will get the following pass^k or pass@k for any k.

| Retail | Airline |
|--------|---------|
| 6.0%   | 38%     |

### Issue 2: A trivial spamming agent scores high pass@k and pass^k

To reproduce the second issue, clone the Tau-Bench repository and apply `exploit-2.patch`
```bash
git clone https://github.com/sierra-research/tau-bench && cd tau-bench
git checkout 14bf0ef52e595922d597a38f32d3e8c0dce3a8f8
git apply ../exploit-2.patch
```

Then, run the spamming agent by
```bash
python run.py \
      --agent-strategy tool-calling \
      --env retail \
      --model gpt-4o \
      --model-provider openai \
      --user-model gpt-4o \
      --user-model-provider openai \
      --user-strategy llm \
      --max-concurrency 10
python run.py \
      --agent-strategy tool-calling \
      --env airline \
      --model gpt-4o \
      --model-provider openai \
      --user-model gpt-4o \
      --user-model-provider openai \
      --user-strategy llm \
      --max-concurrency 10
```

You will get the following pass@k and pass^k for any k:
| Retail | Airline |
|--------|---------|
| 9.6%   | 40%     |
