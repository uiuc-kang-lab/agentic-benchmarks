 <center>
<h1> 
Establishing Best Practices for Rigorous Agentic Benchmarks
</h1>
</center>

<center>
<p>
    Yuxuan Zhu<sup>1</sup>, Tengjun Jin<sup>1</sup>, Yada Pruksachatkun,
    Andy Zhang<sup>2</sup>, Shu Liu<sup>3</sup>, Sasha Cui<sup>4</sup>, 
    Sayash Kapoor<sup>5</sup>, Shayne Longpre<sup>6</sup>, Kevin Meng<sup>7</sup>, 
    Rebecca Weiss<sup>8</sup>, Fazl Barez<sup>8</sup>, Rahul Gupta<sup>9</sup>, 
    Jwala Dhamala<sup>9</sup>, Jacob Merizian<sup>10</sup>, Mario Giulianelli<sup>10</sup>,
    Harry Coppock<sup>10</sup>, Cozmin Ududec<sup>10</sup>, Jasjeet Sekhon<sup>4</sup>, 
    Jacob Steinhardt<sup>7</sup>, Sarah Schwettmann<sup>7</sup>, 
    Matei Zaharia<sup>3</sup>, Ion Stoica<sup>3</sup>, Percy Liang<sup>2</sup>, 
    Daniel Kang<sup>1</sup>
</p>
</center>

<center>
<p>
<sup>1</sup> <img align="center" src="assets/uiuc-full.png" width="100"> &emsp;
<sup>2</sup> <img align="center" src="assets/stanford-full.png" width="100"> &emsp;
<sup>3</sup> <img align="center" src="assets/berkeley-full.png" width="100"> &emsp;
<sup>4</sup> <img align="center" src="assets/yale-full.png" width="60"> &emsp;
<sup>5</sup> <img align="center" src="assets/princeton-full.webp" width="120"> &emsp;
<sup>6</sup> <img align="center" src="assets/mit-full.png" width="60"> &emsp;
</p>

<p>
<sup>7</sup> <img align="center" src="assets/transluce.jpg" width="140"> &emsp;
<sup>8</sup> <img align="center" src="assets/mlcommons.png" width="100"> &emsp;
<sup>9</sup> <img align="center" src="assets/amazon.png" width="100"> &emsp;
<sup>10</sup> <img align="center" src="assets/ukaisi.svg" width="130"> &emsp;
</p>
</center>

<center>
<div class="link-button-group">
  <a href="#" class="pill-button">Paper</a>
  <a href="https://github.com/uiuc-kang-lab/agentic-benchmarks/tree/main" class="pill-button">Repository</a>
  <a href="assets/checklist.pdf" class="pill-button">Checklist</a>
</div>
</center>

## Problem

As AI agents move from research demos to real-world assistants, the only way to 
know what they can (and cannot) do is to test them. Benchmarks have been 
developed as a way to benchmark the high-level capabilities and shortcomings of 
various agentic frameworks and base models, and are crucial in steering research, 
shaping product roadmaps, and helping customers pick the right model.
However, these benchmarks often contain flaws that lead to major misrepresentation 
in performance of up to 40% on popular benchmarks such as SWE-bench-Verified and τ-bench.

## Taxonomy

<center>
<img align="center" src="assets/taxonomy.svg" width="700">
</center>

We identify two major challenges in creating rigorous agentic benchmarks:

1. Task Validity: a task should be solvable if and only if the agent 
    possesses the target capability. 
2. Outcome Validity: the evaluation method (e.g., tests or checks) should 
   indicate correctly whether the task has been solved.

## Checklist Assessment

We develop the [Agentic Benchmark Checklist (ABC)](assets/checklist.pdf), consisting of concrete and
actionable guidelines to ensure outcome and task validity. In cases where 
perfect guarantees of outcome and task validity are particular challenging
or impossible, we also provide 
guidelines to ensure the quality and rigor of benchmark reporting.

We apply ABC on ten widely used agentic benchmarks:

{{ read_csv('assets/scores.csv', colalign=("center", "center", "center", "center", "center",)) }}

Based on our analysis, we suggest the following best practices for benchmark developers:

1. Use process-based evaluation metrics alongside outcome-based metrics. 
2. Benchmark your LLM-as-a-judge in a reproducible manner. Tools such as <a href="https://aligneval.com/">AlignEval</a> can help with evaluating your LLM evaluator.
3. If possible, use frozen websites for tasks that require navigation and website reading.

## Contribute to the Agent Benchmark Checklist
Upholding the validity of agentic benchmarks requires effort from the broader scientific community. If you’re passionate about reliable evaluation in AI, we’d love your help.

Here’s some ways to get involved:

1. Apply the checklist to an existing benchmark - submit [here](https://forms.gle/BRrVh8McQaq8tnGc8).

2. Contribute proof-of-concept exploits and fixes for those exploits in our [repo](https://github.com/uiuc-kang-lab/agentic-benchmarks).

3. Give feedback on the checklist itself [here](https://forms.gle/xbGkqVksEH4fTajF8).