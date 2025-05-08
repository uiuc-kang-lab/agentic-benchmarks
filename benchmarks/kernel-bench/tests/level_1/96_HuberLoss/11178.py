
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Reference implementation using PyTorch's smooth_l1_loss
def smooth_l1_ref(predictions, targets):
    # using the default reduction "mean"
    return torch.nn.functional.smooth_l1_loss(predictions, targets)

# Issue 1: Kernel only supports float32; using other types (e.g. float64) should produce incorrect result.
def test_float32_only():
    # Create double precision tensors on CUDA.
    N = 4096
    pred = torch.randn(128, N, device="cuda", dtype=torch.float64).contiguous()
    targ = torch.randn(128, N, device="cuda", dtype=torch.float64).contiguous()
    kernel = build_kernel()
    # Even though our TORCH_CHECK does not check type explicitly, the kernel will reinterpret
    # the pointer as float*. Therefore, the resulting loss will be off.
    with pytest.raises(Exception) as excinfo:
        # Expecting an error or numerical mismatch because of type confusion.
        out = kernel.forward(pred, targ)
        torch.cuda.synchronize()
    # In case no exception is raised, test that the output is not close to the reference.
    # (Use try/except block to catch both possibilities.)
    try:
        out = kernel.forward(pred, targ)
        torch.cuda.synchronize()
        ref = smooth_l1_ref(pred.float(), targ.float())  # convert to float32 for a fair comparison
        assert not torch.allclose(out, ref, atol=1e-4), \
            "Kernel accepted double precision data but produced values close to the float32 reference."
    except Exception:
        pass

# Issue 2: Misaligned memory access due to vectorized load using float4 without checking alignment.
def test_misaligned_memory():
    # Create a tensor that is contiguous but force non-16-byte alignment by slicing.
    N = 4097  # deliberately choose a size that is not divisible by 4 for the leading dimension
    base_pred = torch.randn(129, N + 1, device="cuda", dtype=torch.float32)  # larger tensor
    base_targ = torch.randn(129, N + 1, device="cuda", dtype=torch.float32)
    # Slice to produce a tensor with an offset of 1 element (4 bytes) in memory.
    pred = base_pred[:, 1:].contiguous()
    targ = base_targ[:, 1:].contiguous()
    # Check that the data_ptr is not 16-byte aligned.
    if (pred.data_ptr() % 16) == 0:
        pytest.skip("Tensor is unexpectedly 16-byte aligned; cannot test misalignment.")
    kernel = build_kernel()
    out = kernel.forward(pred, targ)
    torch.cuda.synchronize()
    ref = smooth_l1_ref(pred, targ)
    # Because of misaligned accesses the kernel may compute an incorrect result.
    assert not torch.allclose(out, ref, atol=1e-5), \
        "Kernel appears to work even with misaligned memory, but it should fail or produce mismatched output."

# Issue 3: Division-by-zero when input tensor is empty.
def test_empty_input():
    # Create empty tensors on CUDA.
    pred = torch.empty(0, device="cuda", dtype=torch.float32)
    targ = torch.empty(0, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(ZeroDivisionError):
        out = kernel.forward(pred, targ)
        # force synchronization to raise potential division by zero error
        torch.cuda.synchronize()

# Issue 4: Kernel assumes CUDA inputs. Passing non‑CUDA tensors should trigger a check.
def test_non_cuda_input():
    # Create CPU tensors.
    pred = torch.randn(128, 4096, device="cpu", dtype=torch.float32)
    targ = torch.randn(128, 4096, device="cpu", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        out = kernel.forward(pred, targ)
        torch.cuda.synchronize()
