# A Challenge Validity Issue in SWE-Lancer

## Overview

SWE-Lancer evaluates the capability of AI agents to implement features and fix 
bugs by granting agents the privilege to view and modify arbitrary files, 
including the test cases. To mitigate cheating SWE-Lancer stored test cases in a 
password-protected .zip file. However, reading the directory structure and 
updating files within a .zip file do not require a password. Therefore, an agent 
can locate test cases and replace them with a naive one, such as “assert 1==1”, 
to achieve a 100% resolve rate without resolving challenges.

To fix the issue, we can zip the test cases twice with passwords.

## Reproducing the Issue

To reproduce the issue, clone the SWE-Lancer repository and apply `exploit.patch`:
```bash
git clone https://github.com/openai/SWELancer-Benchmark.git && cd SWELancer-Benchmark
git checkout 6fee3b0200d90f5b24aab36de6c787ec849e76aa
git apply ../exploit.patch
```

Then, run the SWE-Lancer Benchmark as instructed in SWELancer-Benchmark/README.md

## Fixing the Issue

To fix the issue, clone the SWE-Lancer repository and apply `fix.patch`:
```bash
git clone https://github.com/openai/SWELancer-Benchmark.git && cd SWELancer-Benchmark
git checkout 6fee3b0200d90f5b24aab36de6c787ec849e76aa
git apply ../fix.patch
```

To prove that the issue is fixed, apply `exploit.patch` again and run the benchmark:
```bash
cd SWELancer-Benchmark
git apply ../exploit.patch
# run benchmark as instructured in the README of SWELancer-Benchmark
```