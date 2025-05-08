
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="layernorm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference implementation using PyTorch's layer norm
def ref_layer_norm(x, weight, bias, eps):
    # Assume normalization over the last dimension (flattened)
    normalized_shape = weight.shape
    # Reshape input to (outer, normalized_size)
    outer = x.numel() // torch.prod(torch.tensor(normalized_shape)).item()
    x_reshaped = x.view(outer, -1)
    mean = x_reshaped.mean(dim=1, keepdim=True)
    var = x_reshaped.var(dim=1, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    x_norm = (x_reshaped - mean) * inv_std
    # apply affine transformation elementwise (broadcast weight and bias)
    # weight and bias are assumed to be 1D matching the last dimension
    out = x_norm * weight.view(1, -1) + bias.view(1, -1)
    return out.view_as(x)

# Test 1: Non-contiguous input tensor triggers issue 1
def test_non_contiguous_input():
    my_module = build_kernel()
    batch_size, N, M = 8, 32, 32  # normalized_size = 32*32 = 1024
    # Create a contiguous tensor then make it non-contiguous by transposing dims
    x = torch.randn(batch_size, N, M, device='cuda', dtype=torch.float32)
    x = x.transpose(0, 1)  # now non-contiguous relative to the expected layout
    # weight and bias expected to be contiguous and match normalized dimension size.
    # In a typical LayerNorm the normalized_shape is (batch_size, M)
    # Here we purposely misuse shapes to trigger non-contiguity issues.
    normalized_size = x.numel() // x.size(0)
    weight = torch.randn(normalized_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(normalized_size, device='cuda', dtype=torch.float32)
    
    # Run kernel forward
    out_kernel = my_module.forward(x.contiguous(), weight, bias)
    
    # Compute reference output using reshaping (using the original x before transpose)
    # Because the kernel expects the normalized region to be contiguous,
    # the non-contiguity should lead to a wrong output if it were used without proper checks.
    out_ref = ref_layer_norm(x.contiguous(), weight, bias, 1e-5)
    
    # The test is designed to fail (i.e. find a discrepancy) if the kernel is misbehaving.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel accepted non-contiguous input without error, which may hide compatibility issues."

# Test 2: Using a non power-of-two normalized_size triggers issue 2 (reduction bug)
def test_non_power_of_two_reduction():
    my_module = build_kernel()
    # Choose a normalized_size that is not a power-of-two, e.g. 30 elements.
    batch_size = 4
    normalized_size = 30
    # Create a 2D tensor: outer x normalized_size
    x = torch.randn(batch_size, normalized_size, device='cuda', dtype=torch.float32)
    # weight and bias must match normalized_size.
    weight = torch.randn(normalized_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(normalized_size, device='cuda', dtype=torch.float32)
    
    # Run custom kernel
    out_kernel = my_module.forward(x, weight, bias)
    # Get reference from PyTorch layer norm (simulate over last dim)
    out_ref = ref_layer_norm(x, weight, bias, 1e-5)
    
    # If reduction does not work correctly for non power-of-two sizes,
    # the kernel output will differ notably from the reference.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel reduction appears correct for non power-of-two block sizes, but an issue was expected."

# Test 3: Mismatched affine parameter shapes triggers issue 3
def test_incorrect_weight_bias_shape():
    my_module = build_kernel()
    batch_size, N, M = 8, 16, 16  # normalized_size = 256
    x = torch.randn(batch_size, N, M, device='cuda', dtype=torch.float32)
    # Provide weight and bias of an incorrect shape (e.g. missing one element)
    weight = torch.randn(250, device='cuda', dtype=torch.float32)
    bias = torch.randn(250, device='cuda', dtype=torch.float32)
    
    # The kernel does not perform shape validation; thus, using incorrect shapes
    # may lead to out-of-bound access. We use pytest.raises to catch a RuntimeError
    # or potential CUDA error.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, weight, bias)

# Test 4: Misaligned shared memory allocation may trigger alignment issues (issue 4)
def test_shared_memory_alignment_issue():
    my_module = build_kernel()
    # We attempt to trigger potential alignment related issues by using an input
    # tensor that is deliberately created with a non-standard stride.
    batch_size, normalized_size = 4, 128
    # Create a base tensor and then use as_strided to simulate misalignment.
    base = torch.randn(batch_size * normalized_size + 1, device='cuda', dtype=torch.float32)
    # Create a tensor view that starts at index 1 to force misalignment.
    x = base.narrow(0, 1, batch_size * normalized_size).view(batch_size, normalized_size)
    
    weight = torch.randn(normalized_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(normalized_size, device='cuda', dtype=torch.float32)
    
    # Run the kernel: if alignment issues occur, the kernel result may be incorrect.
    out_kernel = my_module.forward(x, weight, bias)
    out_ref = ref_layer_norm(x, weight, bias, 1e-5)
    
    # We expect a discrepancy because misaligned shared memory accesses may yield wrong results.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel produced correct results for misaligned memory input, which is unexpected given our assumptions."
