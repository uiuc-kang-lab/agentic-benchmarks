
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Compile the CUDA extension from the file kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def get_forward(mod, input, weight, bias, stride, padding, output_padding, groups):
    # Call the forward function from the loaded module.
    return mod.forward(input, weight, bias, stride, padding, output_padding, groups)

def test_non_float_dtype():
    # Issue 2: The kernel does not check for the correct data type.
    # Create float64 tensors instead of float32, and expect the kernel to fail.
    mod = build_kernel()
    # Prepare a simple input in float64 on CUDA.
    input = torch.randn(2, 4, 5, 5, 5, dtype=torch.float64, device="cuda")
    # Create weight with the expected shape [C_in, C_out/groups, kD, kH, kW]
    weight = torch.randn(4, 2, 3, 3, 3, dtype=torch.float64, device="cuda")
    # Create bias if needed (None here)
    bias = None
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    output_padding = [0, 0, 0]
    groups = 1

    with pytest.raises(RuntimeError):
        # This call should fail because the kernel expects float32.
        _ = get_forward(mod, input, weight, bias, stride, padding, output_padding, groups)

def test_in_channels_not_divisible_by_groups():
    # Issue 3: The kernel assumes in_channels is divisible by groups.
    # We'll intentionally create an input tensor where the number of channels is not divisible by groups.
    mod = build_kernel()
    # For example, 5 input channels with groups=2 (5 % 2 != 0)
    N, C_in, D, H, W = 2, 5, 4, 4, 4
    # Weight shape: [C_in, C_out_per_group, kD, kH, kW] where C_out will be groups * C_out_per_group.
    # We choose arbitrary kernel dimensions.
    kD, kH, kW = 3, 3, 3
    # Set output channels arbitrarily, e.g., 6 (so C_out_per_group should be 3, but groups=2 and C_in=5 is invalid)
    C_out = 6  
    groups = 2  # 5 is not divisible by 2.
    
    input = torch.randn(N, C_in, D, H, W, dtype=torch.float32, device="cuda")
    weight = torch.randn(C_in, C_out // groups, kD, kH, kW, dtype=torch.float32, device="cuda")
    bias = None
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    
    # We expect the kernel execution to fail (e.g., due to out-of-bound memory access) or at least produce a runtime error.
    with pytest.raises(Exception):
        _ = get_forward(mod, input, weight, bias, stride, padding, output_padding, groups)

def test_unused_shared_memory():
    # Issue 1: The kernel allocates shared memory that remains unused.
    # Although this does not affect numerical correctness directly, we include a dummy test that exercises the kernel.
    mod = build_kernel()
    N, C_in, D, H, W = 2, 4, 4, 4, 4
    kD, kH, kW = 3, 3, 3
    C_out = 4
    groups = 1
    input = torch.randn(N, C_in, D, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out // groups, kD, kH, kW, device="cuda", dtype=torch.float32)
    bias = None
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    
    out = get_forward(mod, input, weight, bias, stride, padding, output_padding, groups)
    # Since there is no reference output for this placeholder test, we only assert that an output is produced.
    assert out is not None

def test_warp_level_primitives_documentation():
    # Issue 4: The kernel’s comment mentions warp-level primitives but none are used.
    # Since runtime detection is not possible, we perform a placeholder test that checks for the phrase in the source file.
    # This test reads the kernel.cu file and searches for "warp-level".
    try:
        with open("kernel.cu", "r") as f:
            contents = f.read()
    except FileNotFoundError:
        pytest.skip("kernel.cu file not found")
    
    # Check that the file contains the phrase "warp-level primitives" (documented intent) but note that no such
    # primitives are actually implemented.
    assert "warp-level primitives" in contents, \
        "Kernel documentation mentions warp-level primitives; however, the implementation does not use any."
