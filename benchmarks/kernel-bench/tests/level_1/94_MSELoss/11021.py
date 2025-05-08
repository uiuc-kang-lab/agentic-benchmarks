
import torch
import pytest
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

# Issue 1: Hard-coded SM count
# While we cannot change the device properties in a test,
# we can check that the kernel produces correct results even when
# the hardware differs from the assumed "108 SM" configuration.
def test_hardcoded_sm_count():
    my_module = build_kernel()
    # Create a sufficiently large tensor to force multiple blocks.
    N = 4096 * 20 + 3  # deliberately not a multiple of 4 to also hit issue 3
    preds = torch.randn(N, device="cuda", dtype=torch.float32)
    tgts = torch.randn(N, device="cuda", dtype=torch.float32)
    
    # Compute reference MSE using PyTorch
    ref = torch.mean((preds - tgts) ** 2)
    # Compute using the CUDA kernel
    out = my_module.forward(preds, tgts)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), (
        f"Kernel output {out.item()} differs from reference {ref.item()} on device with SM count mismatches."
    )

# Issue 2: Block-size assumption in the reduction unrolling.
# We simulate this by using a tensor whose total number of elements
# forces reduction across multiple blocks and boundary conditions.
def test_reduction_blocksize_assumption():
    my_module = build_kernel()
    # Create a tensor with a number of elements that is not a multiple of (BLOCK_SIZE*VECTOR_SIZE).
    N = 1024 + 7  # small non-uniform size to exercise boundary conditions in reduction.
    preds = torch.randn(N, device="cuda", dtype=torch.float32)
    tgts = torch.randn(N, device="cuda", dtype=torch.float32)
    
    ref = torch.mean((preds - tgts) ** 2)
    out = my_module.forward(preds, tgts)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), (
        f"Reduction unrolling issue: Kernel output {out.item()} differs from reference {ref.item()}."
    )

# Issue 3: Vectorized load misalignment causing divergence when number of elements is not a multiple of VECTOR_SIZE.
def test_non_multiple_of_vector_size():
    my_module = build_kernel()
    # Create inputs such that num_elements % VECTOR_SIZE != 0 (with VECTOR_SIZE==4)
    N = 4096 + 2  # deliberately create a remainder when dividing by 4
    preds = torch.randn(N, device="cuda", dtype=torch.float32)
    tgts = torch.randn(N, device="cuda", dtype=torch.float32)
    
    ref = torch.mean((preds - tgts) ** 2)
    out = my_module.forward(preds, tgts)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), (
        f"Vectorized load misalignment issue: Kernel output {out.item()} differs from reference {ref.item()}."
    )

# Issue 4: Lack of error checking after the kernel launch.
# We can simulate misuse that should trigger TORCH_CHECK errors by providing wrong device tensors.
def test_kernel_no_error_checking_in_launch():
    my_module = build_kernel()
    N = 1024
    preds = torch.randn(N, device="cpu", dtype=torch.float32)  # CPU tensor instead of CUDA
    tgts = torch.randn(N, device="cpu", dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="predictions must be a CUDA tensor"):
        my_module.forward(preds, tgts)

# Issue 5: atomicAdd on double is hardware-dependent.
# We simulate a possible misuse by trying to call the kernel with double-precision inputs.
# Most modern GPUs support atomicAdd on double, so to test the limitation, we force
# an input of type double and expect the operation to complete; however, if the underlying
# GPU does not support it, an error would be raised. We check that the function either
# produces a result close to reference or fails with a clear error message.
def test_atomicAdd_double_support():
    my_module = build_kernel()
    N = 4096
    # Use double precision input to trigger the atomicAdd on double code path.
    preds = torch.randn(N, device="cuda", dtype=torch.float64)
    tgts = torch.randn(N, device="cuda", dtype=torch.float64)
    
    ref = torch.mean((preds - tgts) ** 2)
    try:
        out = my_module.forward(preds, tgts)
        torch.cuda.synchronize()
        # Convert out to double for comparison if needed.
        out = out.to(torch.float64)
        assert torch.allclose(out, ref, atol=1e-5), (
            f"AtomicAdd double issue: Kernel output {out.item()} differs from reference {ref.item()}."
        )
    except RuntimeError as e:
        msg = str(e)
        assert "atomicAdd" in msg or "not supported" in msg, (
            "Expected an error related to atomicAdd on double but got a different error."
        )
