
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension from kernel.cu.
    # Note: If there is a compilation error (such as the undefined "min" issue),
    # this function will raise an error.
    cuda_module = load(
        name="gelu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_float_tensor():
    """
    Test Issue 2: The kernel does not check for the tensor type.
    Supply a double tensor (float64) to trigger the unexpected behavior.
    """
    cuda_module = build_kernel()
    # Create a tensor in double precision on CUDA.
    x = torch.randn(1024, dtype=torch.float64, device="cuda")
    # Running the kernel on a non-float tensor (float64) is not explicitly prevented.
    # It may produce incorrect results. We expect a failure or significant numerical mismatch.
    with pytest.raises(RuntimeError):
        # Since the kernel expects float pointer, the misinterpreted memory may raise an error.
        y = cuda_module.forward(x)
        torch.cuda.synchronize()  # force any asynchronous errors

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_tensor():
    """
    Test Issue 2 (also related to inputs expected by the kernel): 
    Passing a noncontiguous tensor should trigger the contiguous check.
    """
    cuda_module = build_kernel()
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes the tensor non-contiguous
    with pytest.raises(RuntimeError, match="contiguous"):
        y = cuda_module.forward(x_noncontig)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_not_on_cuda():
    """
    Test Issue 2 (input validation): Passing a CPU tensor should trigger a check.
    """
    cuda_module = build_kernel()
    x = torch.randn(1024, 1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="on CUDA"):
        y = cuda_module.forward(x)

# Note: The "min" issue (Issue 1) is a compile‐time error.
# If the extension is built successfully, then the "min" issue is no longer present.
# To simulate this issue in a test, one would need to check the build logs.
# We can include a dummy test that fails if the module does not raise during build,
# but typically the build step itself will error out if "min" is undefined.

def test_build_failure_for_min_issue():
    """
    Dummy test to indicate that if the extension build passes, the "min" issue was not caught.
    In real usage, a failure to compile due to the undefined 'min' function would prevent the tests from running.
    """
    try:
        cuda_module = build_kernel()
    except Exception as e:
        err = str(e)
        assert "min" in err or "std::min" in err, (
            "Expected compilation error related to undefined 'min', but got a different error."
        )
    else:
        # If the module built successfully, we note that the min issue did not occur,
        # which may indicate that corrections (or different headers) were added.
        pytest.skip("Module built successfully. 'min' issue not triggered.")
