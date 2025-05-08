# Outcome Validity Issues in Kernel-Bench

## Overview

Kernel-Bench uses randomly generated tensors to test the correctness of 
agent generated CUDA kernel functions. However, the fuzzing strategy only 
changes the values in the input tensor. This leads to false positive kernel 
functions that fail on tensors with different shapes and memory layouts.

## Reproducing Issues with Prior Public Generations

We use previous generations by Sakana-AI to demonstrate outcome validity issues 
in Kernel-Bench. It takes three steps to reproduce our results.

1. Ensure dependency is installed
```bash
# make sure python 3.11.11 and CUDA 12.4 are installed
pip install -r requirements.txt
```

2. Collect and preprocess the generation data by Sakana-AI. This will also 
save the generated kernels to `kernels` and sample a set of generations that 
were marked as "correct" for test case generation.

```bash
python scripts/preprocess_data.py
```

3. Use LLM to generate more test cases. 

```bash
export OPENAI_API_KEY=
python scripts/judge_by_llm.py
```
Generated test cases will be saved in `./tests/`. We recommending manually 
verify these test cases for correctness. We've provided both the original 
generated test cases in `./tests` and our verified test cases in 
`./tests_verified`.

4. Run verified tests on generated kernels.

```bash
# for each kernel under ./kernels
export KERNEL_ID=
export REPORT_FILE=
export TEST_FILE=
MAX_JOBS=4 pytest --timeout=120 --json-report --json-report-file=$REPORT_FILE $TEST_FILE > logs/$KERNEL_ID.log 2>&1
```