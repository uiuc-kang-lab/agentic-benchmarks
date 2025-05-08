
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to build the CUDA extension.
def build_kernel(extra_cuda_cflags=None):
    src_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="cosine_similarity_loss_module",
        sources=[src_file],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1 and 2: Test with D larger than blockDim.x such that not all data is loaded,
# and with a dimension small enough to potentially reveal the shared memory overlap.
@pytest.mark.parametrize("D", [64, 4096])
def test_incorrect_shared_memory_usage(D):
    # Use a batch size of 8 for testing; one block per sample.
    N = 8
    # Create inputs with known patterns.
    # Use a constant value so that the cosine similarity should be 1 (loss 0) when predictions==targets.
    predictions = torch.full((N, D), 1.0, device="cuda", dtype=torch.float32)
    targets = torch.full((N, D), 1.0, device="cuda", dtype=torch.float32)
    
    # Load our kernel module.
    module = build_kernel()
    loss = module.forward(predictions, targets)
    torch.cuda.synchronize()
    # Expected loss: cosine similarity should be 1 for identical inputs => loss = 0.
    expected_loss = 0.0

    # If shared memory is used correctly, loss should be near 0.
    # Due to improper data loading or memory overlapping, the result might be far off.
    # Here we trigger the issue by asserting that the loss is not close to 0.
    assert not torch.isclose(loss, torch.tensor(expected_loss, device="cuda"), atol=1e-3).item(), (
        "Kernel produced correct loss while an error was expected due to shared memory misuse."
    )

# Issue 3: Test that the kernel fails to compile if warpSize is undefined.
def test_undefined_warpSize():
    # We simulate the scenario by not defining warpSize.
    # In a correct build environment, warpSize (32) is standard; however, our kernel uses it without definition.
    # We force a build with an extra flag that undefines warpSize.
    extra_flags = ["-O3", "--use_fast_math", "-UwarpSize"]
    with pytest.raises(RuntimeError):
        build_kernel(extra_cuda_cflags=extra_flags)

# Issue 4: Test with an extremely large D that might trigger shared memory allocation failure.
def test_shared_memory_overflow():
    # Many GPUs have a limit on dynamic shared memory (often around 48K bytes per block).
    # Choose D large enough so that (2*D + 3*numWarps)*sizeof(float) exceeds the device limit.
    # For instance, if block_size is 256, numWarps = 8, so total shared memory in bytes = (2*D + 24)*4.
    # Setting D to 10000 gives about 80024 bytes, likely to exceed the limit on many devices.
    N = 4
    D = 10000
    predictions = torch.randn(N, D, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, D, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(predictions, targets)
    torch.cuda.synchronize()

# Additional test: Verify that incorrect input tensor type is caught.
def test_input_tensor_type():
    N = 8
    D = 128
    # Create input tensors with type double rather than float32.
    predictions = torch.randn(N, D, device="cuda", dtype=torch.float64)
    targets = torch.randn(N, D, device="cuda", dtype=torch.float64)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(predictions, targets)
    torch.cuda.synchronize()
