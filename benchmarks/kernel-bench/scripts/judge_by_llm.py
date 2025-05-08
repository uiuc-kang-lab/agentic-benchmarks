prompt_template = """\
Act as a expert CUDA kernel engineer and identify all the issues in a new implementation of a CUDA kernel for a PyTorch function.
Here is the potential PyTorch code where the new CUDA kernel will be used. But Keep in mind that the new CUDA kernel should be compatible with other more complex and general situations.
```
{PyTorch_Code_Module}
```

Here is a new CUDA kernel at the file `kernel.cu`:
```c++
{CUDA_Code}
```

If there are issues in the new CUDA kernel, please list them in detail. \
In addition to the list, please provide a pytest file containing one test caes for each issue. \
Here are the requirements for the pytest file: \
1. The test cases should be designed to trigger the issues in the CUDA kernel.
2. The test cases should be written in Python and should be compatible with pytest.
3. The test cases should test the kernel function in a file called `kernel.cu`.
Please wrap the test cases in a code block. \

If there are no issues, respond with "No issues found".

Here is two examples of the output format. Please follow the format strictly.

Example 1:
No issues found.

Example 2:
Issues:
1. The kernel does not handle the case when the input tensor is not of type float32.
2. The kernel does not handle the case when the dimensions of input tensor are not divisible by 32.

Pytest file:
```python
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_input_tensor_type():
    N = 1024
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Difference: {{(C-C_ref).abs().max()}}"

def test_input_tensor_dim():
    N = 70
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Difference: {{(C-C_ref).abs().max()}}"

```
"""

from openai import OpenAI
import pandas as pd
import os, json, sys

def parse_llm_output(response):
    if "```python" in response:
        tests = response.split("```python")[1]
        tests = tests.split("```")[0]
        return [response, tests]
    else:
        return [response, None]

def get_llm_response(prompt, client):
    response = client.responses.create(
        model="o3-mini",
        input=prompt
        )
    cost = response.usage.input_tokens / 1e6 * 1.1 + response.usage.output_tokens / 1e6 * 4.4
    return response.output[1].content[0].text, response.usage.input_tokens, response.usage.output_tokens, cost

def generate_test_cases(dataset: pd.DataFrame):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    running_cost = 0
    for index, row in dataset.iterrows():
        if row["Correct"] == True:
            prompt = prompt_template.format(
                PyTorch_Code_Module=row["PyTorch_Code_Module"],
                CUDA_Code=row["CUDA_Code"]
            )
            response, n_input, n_output, cost = get_llm_response(prompt, client)
            response, code = parse_llm_output(response)
            if code:
                os.makedirs(f"tests/level_{row['Level_ID']}/{row['Op_Name']}", exist_ok=True)
                with open(f"tests/level_{row['Level_ID']}/{row['Op_Name']}/{row['id']}.py", "w") as f:
                    f.write(code)
            with open(f"llm_judge_{level_id}.jsonl", "a+") as f:
                result = {
                    "Level_ID": row["Level_ID"],
                    "id": row["id"],
                    "judge": response,
                    "input_tokens": n_input,
                    "output_tokens": n_output,
                    "cost": cost,
                    "valid_tests": code is not None,
                }
                f.write(json.dumps(result) + "\n")
            running_cost += cost
            print(f"Processed {index + 1}/{len(dataset)}: {row['Op_Name']} - {row['id']}. Cost: {running_cost:.2f} USD")

level_id = sys.argv[1]

data = pd.read_parquet(f"data/level_{level_id}_sampled_data.parquet")
generate_test_cases(data)