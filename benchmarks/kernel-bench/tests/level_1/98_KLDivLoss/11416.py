
import os
import re
import tempfile
import torch
import pytest
from torch.utils.cpp_extension import load

# A helper function to load the CUDA extension from a given source string.
def load_extension_from_source(source, extra_cuda_cflags=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.cu")
        with open(src_path, "w") as f:
            f.write(source)
        module = load(
            name="custom_kernel",
            sources=[src_path],
            extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math"],
            verbose=False,
        )
    return module

# Read the original kernel source from kernel.cu
with open("kernel.cu", "r") as f:
    original_source = f.read()

############################################
# Test case for Issue 1: Incorrect KL divergence formula.
############################################
def test_incorrect_formula():
    # Use the original kernel source.
    module = load_extension_from_source(original_source)
    
    # Create inputs.
    # We use small sizes so the kernel reduction is easier to analyze.
    batch_size = 4
    num_features = 10  # making a small vector for each sample.
    # Create valid probability distributions.
    predictions = torch.randn((batch_size, num_features), device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn((batch_size, num_features), device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    # Compute our CUDA kernel output.
    # Note: The extension returns a tensor that is output/n.
    kernel_output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    
    # Compute expected result using PyTorch's F.kl_div.
    # F.kl_div expects inputs = log(predictions), targets:
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")
    
    # Because the kernel formula is different, the difference should be significant.
    diff = (kernel_output - expected).abs().item()
    # We set a threshold that the kernel result must differ from expected.
    assert diff > 1e-3, f"Kernel formula issue not detected. Diff = {diff}"

############################################
# Test case for Issue 2: Incorrect normalization (dividing by n instead of batch size).
############################################
def test_incorrect_normalization():
    module = load_extension_from_source(original_source)
    
    # Create inputs with a multi-dimensional shape so that total elements n != batch_size.
    batch_size = 8
    num_features = 32  # n = 8*32 = 256, so if normalized by n, divisor would be 256 instead of 8.
    
    predictions = torch.randn((batch_size, num_features), device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn((batch_size, num_features), device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    kernel_output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    
    # Expected using torch.nn.functional.kl_div normalized by batch size.
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")
    
    # The kernel divides by n (256) instead of batch_size (8), so the result will be off by a factor of 32.
    ratio = kernel_output.item() / expected.item() if expected.item() != 0 else 0.0
    assert abs(ratio - 1/32.0) < 1e-2, f"Normalization issue: expected ratio ~{1/32.0}, got {ratio}"

############################################
# Test case for Issue 3: Shared memory size hard-coded to 8 warps.
############################################
def test_shared_memory_assumption():
    # Modify the kernel source to launch with 512 threads per block instead of 256.
    # This will simulate a scenario where blockDim.x/32 = 16 warps
    modified_source = re.sub(r"const int threads = 256;", "const int threads = 512;", original_source)
    module = load_extension_from_source(modified_source)
    
    # Create inputs with enough elements to launch at least one block with 512 threads.
    n_elements = 1024  # arbitrary number > 512 so that one block is used
    predictions = torch.randn(n_elements, device="cuda", dtype=torch.float32).softmax(dim=0)
    targets = torch.randn(n_elements, device="cuda", dtype=torch.float32).softmax(dim=0)
    
    kernel_output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    
    # Compute expected result using PyTorch's F.kl_div over flattened tensors.
    # We emulate batchmean by using batch size 1 if needed.
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")
    
    # Because shared memory is written out-of-bounds, the kernel output will likely be corrupted.
    diff = (kernel_output - expected).abs().item()
    # We expect a significant difference.
    assert diff > 1e-3, f"Shared memory issue not detected; diff = {diff}"

############################################
# Test case for Issue 4: Input tensor type not enforced (float32 expected).
############################################
def test_incorrect_tensor_type():
    module = load_extension_from_source(original_source)
    
    # Create inputs with double precision.
    batch_size = 4
    num_features = 10
    predictions = torch.randn((batch_size, num_features), device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn((batch_size, num_features), device="cuda", dtype=torch.float64).softmax(dim=-1)
    
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel expects float (float32).
        _ = module.forward(torch.log(predictions), targets)
