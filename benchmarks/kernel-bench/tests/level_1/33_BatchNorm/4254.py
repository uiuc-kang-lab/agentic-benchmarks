
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile the CUDA extension
def build_kernel():
    cuda_module = load(
        name="batchnorm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper function to call the forward function from the extension.
# The forward function implements BatchNorm using our three kernels.
def run_forward(input, weight, bias, running_mean, running_var, training, momentum, eps):
    cuda_mod = build_kernel()
    # call forward from the loaded CUDA extension
    out = cuda_mod.forward(input, weight, bias, running_mean, running_var, training, momentum, eps)
    torch.cuda.synchronize()
    return out

# Issue 1: Kernel supports only float32 inputs.
def test_dtype_support():
    # Create input and parameters in double precision.
    input = torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.float64)
    weight = torch.randn(64, device="cuda", dtype=torch.float64)
    bias = torch.randn(64, device="cuda", dtype=torch.float64)
    running_mean = torch.zeros(64, device="cuda", dtype=torch.float64)
    running_var = torch.ones(64, device="cuda", dtype=torch.float64)
    momentum = 0.1
    eps = 1e-5

    with pytest.raises(RuntimeError) as excinfo:
        _ = run_forward(input, weight, bias, running_mean, running_var, True, momentum, eps)
    assert "not a CUDA tensor" not in str(excinfo.value), "Expected failure due to data type mismatch, not CUDA placement."

# Issue 2: Kernel is hard-coded for 4D inputs.
def test_input_dimensionality():
    # Create a 3D tensor (which is a valid input for BatchNorm1d, but our kernel expects 4D)
    input = torch.randn(16, 64, 256, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)
    bias = torch.randn(64, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(64, device="cuda", dtype=torch.float32)
    running_var = torch.ones(64, device="cuda", dtype=torch.float32)
    momentum = 0.1
    eps = 1e-5

    with pytest.raises(IndexError) as excinfo:
        _ = run_forward(input, weight, bias, running_mean, running_var, True, momentum, eps)
    # The error may be an IndexError because indexing (H, W) is not valid.
    assert "index" in str(excinfo.value).lower()

# Issue 3: Reduction kernels assume power-of-two block sizes.
# We simulate a scenario that could reveal a reduction issue by forcing an input shape
# such that the total number of elements per channel is not a multiple of a power-of-two.
def test_non_power_of_two_reduction():
    # Using a shape that gives a total number of channel elements not divisible by 256.
    # For example: 1 x 1 x 7 x 7 = 49 elements per channel.
    input = torch.randn(1, 1, 7, 7, device="cuda", dtype=torch.float32)
    weight = torch.ones(1, device="cuda", dtype=torch.float32)
    bias = torch.zeros(1, device="cuda", dtype=torch.float32)
    # Initialize running statistics with non-zero values to help detect errors.
    running_mean = torch.full((1,), 10.0, device="cuda", dtype=torch.float32)
    running_var = torch.full((1,), 10.0, device="cuda", dtype=torch.float32)
    momentum = 0.1
    eps = 1e-5

    # Run in training mode to force the reduction in kernel 2.
    output = run_forward(input, weight, bias, running_mean.clone(), running_var.clone(), True, momentum, eps)
    # Compute expected mean and variance using PyTorch (manual computation):
    channel_elems = input.numel()  # 49
    expected_mean = float(input.sum() / channel_elems)
    expected_var = float(input.pow(2).sum() / channel_elems - expected_mean**2)

    # As the reduction is non-power-of-two in size, we expect relative error.
    # Here we simply check that the normalized output is not equal to what would be computed with correct moments.
    inv_std = 1.0 / ((expected_var + eps) ** 0.5)
    expected_output = (input - expected_mean) * inv_std * weight[0] + bias[0]
    # The output may be off significantly if the reduction is done incorrectly.
    assert not torch.allclose(output, expected_output, atol=1e-4), \
        "Reduction kernel appears to handle non-power-of-two sizes correctly; expected an error scenario."

# Issue 4: Inference mode still performs full reduction unnecessarily.
def test_inference_mode_efficiency():
    # When training is False, the kernel should use running statistics as-is.
    # We set running_mean and running_var to known values.
    input = torch.randn(16, 64, 32, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)
    bias = torch.randn(64, device="cuda", dtype=torch.float32)
    # Setting running mean/var to specific nonzero values.
    running_mean = torch.randn(64, device="cuda", dtype=torch.float32)
    running_var = torch.abs(torch.randn(64, device="cuda", dtype=torch.float32))  # variance must be nonnegative
    running_mean_before = running_mean.clone()
    running_var_before = running_var.clone()
    momentum = 0.1
    eps = 1e-5

    output = run_forward(input, weight, bias, running_mean, running_var, False, momentum, eps)
    # In inference mode, running stats should not update. If they change, then unnecessary reduction has corrupted them.
    assert torch.allclose(running_mean, running_mean_before), "running_mean was modified in inference mode."
    assert torch.allclose(running_var, running_var_before), "running_var was modified in inference mode."

# Issue 5: Kernel enforces contiguous tensor layouts.
def test_input_contiguity():
    input = torch.randn(16, 64, 32, 32, device="cuda", dtype=torch.float32)
    # Make the tensor non-contiguous by transposing some dimensions.
    input_non_contig = input.transpose(2, 3)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)
    bias = torch.randn(64, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(64, device="cuda", dtype=torch.float32)
    running_var = torch.ones(64, device="cuda", dtype=torch.float32)
    momentum = 0.1
    eps = 1e-5

    with pytest.raises(RuntimeError) as excinfo:
        _ = run_forward(input_non_contig, weight, bias, running_mean, running_var, True, momentum, eps)
    assert "contiguous" in str(excinfo.value).lower(), "Expected error for non-contiguous tensor input."
