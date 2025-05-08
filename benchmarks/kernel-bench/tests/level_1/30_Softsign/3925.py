
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from the kernel.cu file.
    cuda_module = load(
        name="softsign_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to compare the softsign operation
def torch_softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.abs(x))

# Test 1. Invalid dtype (Issue 1)
def test_invalid_dtype():
    my_module = build_kernel()
    # Create a tensor with type double; note that our kernel always treats data as float32.
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    # Compute reference using torch_softsign but cast to double so that they match type.
    ref = torch_softsign(x)
    # The kernel call will reinterpret the double bit pattern as float32.
    out = my_module.forward(x)
    # Because of the type mismatch, the output should differ significantly from the reference.
    # We expect that the maximum difference is very high.
    diff = (out.double() - ref).abs().max().item()
    assert diff > 1e-3, f"Expected significant difference due to incorrect type handling, got diff={diff}"

# Test 2. Missing CUDA error checking after launch (Issue 2)
# We simulate an error (for example by passing an extremely large tensor size that might overflow)
def test_kernel_launch_error():
    my_module = build_kernel()
    # Use a very large number of elements that might cause resources issues.
    # Note: This test may or may not trigger an error depending on your GPU,
    # but it is intended to expose that errors from kernel launch are not caught.
    # We use a try-except block to capture a potential CUDA error.
    try:
        # Create a tensor with a huge number of elements to stress the kernel.
        # Here, the intent is to force an error; however, if your GPU can handle it,
        # the test will pass. Adjust the size if needed to trigger error in your environment.
        num_elements = 2**30  # around 1 billion elements
        x = torch.randn(num_elements, device="cuda", dtype=torch.float32)
        out = my_module.forward(x)
        # Force synchronization to catch any asynchronous errors.
        torch.cuda.synchronize()
    except RuntimeError as e:
        # If an error is caught, ensure that it is due to a CUDA runtime error.
        assert "CUDA" in str(e)
    else:
        # If no error is raised, warn that error checking is not triggered.
        pytest.skip("Kernel launch error not triggered; environment may allow huge launches.")

# Test 3. Unqualified use of min causing compile issues (Issue 3)
# While this issue is caught at compile time, we add a basic test to check that the kernel is built and runs.
def test_kernel_compilation_and_execution():
    my_module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    out = my_module.forward(x)
    torch.cuda.synchronize()
    ref = torch_softsign(x)
    # Verify that the output is close to the expected result.
    assert torch.allclose(out, ref, atol=1e-5), "Kernel output does not match expected softsign result."

# Test 4. Non-contiguous tensor input should raise an error due to the CHECK_CONTIGUOUS macro.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous with a transpose.
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    x_non_contig = x.t()  # transpose makes it non-contiguous
    with pytest.raises(RuntimeError, match="must be contiguous"):
        my_module.forward(x_non_contig)

# Test 5. CPU tensor input should raise an error due to the CHECK_CUDA macro.
def test_cpu_tensor_input():
    my_module = build_kernel()
    # Create a CPU tensor
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        my_module.forward(x_cpu)
