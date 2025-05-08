
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Missing std::min qualification / missing <algorithm> include.
# Test: the module should fail to compile if min is not properly qualified.
# We attempt to build the module and expect a compile-time error.
# (If the build succeeds, then no compilation error is noted; this test is to flag an issue if the build unexpectedly passes.)
def test_missing_min_qualification():
    try:
        module = build_kernel()
    except Exception as e:
        # We expect a compilation error mentioning min or similar
        err_msg = str(e)
        assert "min" in err_msg or "std::min" in err_msg, (
            "Compilation error did not mention min usage; expected issue with unqualified min."
        )
    else:
        pytest.skip("Kernel compiled successfully; test for unqualified min not triggered on this system.")

# Issue 2: Using input.type() instead of input.scalar_type() in AT_DISPATCH.
# Test: Pass a double tensor (torch.float64) which may be dispatched incorrectly.
def test_dispatch_input_type():
    module = build_kernel()
    # Create a double-precision tensor on CUDA.
    x = torch.randn(1000, device="cuda", dtype=torch.float64)
    # Compute softplus using the kernel.
    y_kernel = module.forward(x)
    # Compute reference softplus using PyTorch built-in.
    y_ref = torch.nn.functional.softplus(x)
    # If the dispatch is done with wrong type info, the computed result might be off.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Kernel output does not match reference softplus for double precision input. "
        "This may indicate an issue with AT_DISPATCH using input.type() instead of input.scalar_type()."
    )

# Issue 3: Lack of error checking after kernel launch.
# Test: We purposely trigger an error by sending a CPU tensor instead of CUDA tensor.
def test_kernel_launch_error():
    module = build_kernel()
    x_cpu = torch.randn(1000)  # CPU tensor; kernel expects CUDA
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(x_cpu)
    assert "CUDA" in str(excinfo.value) or "device" in str(excinfo.value), (
        "Expected a RuntimeError due to passing a CPU tensor, indicating lack of proper kernel launch error check."
    )

# Issue 4: Device-specific launch parameters.
# Test: Run the kernel on a tensor whose size forces a very low number of blocks.
# Although the kernel may still compute correctly, this test draws attention to the fixed
# launch configuration which might be suboptimal on devices other than H100.
def test_launch_configuration_on_small_tensor():
    module = build_kernel()
    # Create a tiny tensor so that size/threads < sm_count * blocks_per_sm.
    x = torch.randn(10, device="cuda", dtype=torch.float32)
    y_kernel = module.forward(x)
    y_ref = torch.nn.functional.softplus(x)
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Kernel output does not match reference on a small tensor; check launch configuration parameters."
    )

# Issue 5: Kernel interface expectation mismatch.
# Test: Attempt to call the kernel with a second argument (simulating the provided example mismatch).
def test_invalid_kernel_signature():
    module = build_kernel()
    x = torch.randn(100, device="cuda")
    # The kernel expects one input tensor. Passing an extra parameter should cause an error.
    with pytest.raises(TypeError) as excinfo:
        # Intentionally passing a second argument.
        module.forward(x, x)
    assert "forward() takes" in str(excinfo.value), (
        "Kernel forward function did not raise error for invalid number of arguments."
    )
