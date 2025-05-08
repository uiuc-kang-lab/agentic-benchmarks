
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    module = build_kernel()
    return module

def test_non_contiguous_input(cuda_module):
    # Create a contiguous input first and then make it non-contiguous.
    batch_size, features, dim1, dim2 = 4, 8, 16, 16
    # normalized shape covers the last three dims (features, dim1, dim2)
    x = torch.randn(batch_size, features, dim1, dim2, device="cuda")
    # Make a non-contiguous tensor by transposing dimensions (swap two dims)
    x_noncontig = x.transpose(1, 2)
    # Create weight and bias of shape = (features*dim1*dim2) to match default assumption.
    normalized_size = features * dim1 * dim2
    weight = torch.ones(normalized_size, device="cuda", dtype=x.dtype)
    bias = torch.zeros(normalized_size, device="cuda", dtype=x.dtype)
    
    # This test triggers issue 1: input non contiguous, but the kernel does a simple pointer arithmetic.
    with pytest.raises(RuntimeError):
        # The kernel’s arithmetic is based on contiguous layout. This call may produce wrong results or crash.
        _ = cuda_module.forward(x_noncontig, weight, bias)

def test_weight_bias_broadcast(cuda_module):
    # Create a contiguous input tensor with shape [batch, normalized_size]
    batch_size = 8
    normalized_shape = (32,)
    normalized_size = np.prod(normalized_shape)
    x = torch.randn(batch_size, normalized_size, device="cuda")
    
    # Create weight and bias which are broadcastable (e.g., 1D tensor of length 1) but not of length normalized_size.
    weight = torch.tensor([1.0], device="cuda", dtype=x.dtype)
    bias = torch.tensor([0.0], device="cuda", dtype=x.dtype)
    
    # This should trigger issue 2: the kernel expects weight and bias to have exactly normalized_size elements.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x, weight, bias)

def test_noncontiguous_normalized_dims(cuda_module):
    # Create a 4D tensor but then force a situation where the normalized dimensions are not contiguous.
    # Suppose we use input shape (batch, features, dim1, dim2) and normalized shape covering (features, dim1, dim2).
    batch_size, features, dim1, dim2 = 2, 4, 8, 8
    x = torch.randn(batch_size, features, dim1, dim2, device="cuda")
    # Permute dims so that the normalized dimensions are scattered.
    x_perm = x.permute(0, 2, 1, 3).contiguous()  # Even if made contiguous here, the effective layout differs.
    # Now break contiguity along the normalized dimension by slicing
    x_broken = x_perm[:, :, ::2, :]  # This makes the normalized block irregular.
    # We expect the kernel to compute wrong results if it uses linear offset arithmetic.
    normalized_size = x_broken.size(1) * x_broken.size(2) * x_broken.size(3)
    weight = torch.ones(normalized_size, device="cuda", dtype=x.dtype)
    bias = torch.zeros(normalized_size, device="cuda", dtype=x.dtype)
    
    # Since the kernel assumes contiguous normalized dimensions, we wrap the call expecting a wrong result.
    y = cuda_module.forward(x_broken, weight, bias)
    # We compute a naive reference LayerNorm using PyTorch’s function on the flattened last dims.
    x_broken_flat = x_broken.view(-1, normalized_size)
    ref = torch.nn.functional.layer_norm(x_broken_flat, (normalized_size,), weight.view(normalized_size), bias.view(normalized_size), eps=1e-5)
    ref = ref.view_as(x_broken)
    
    # The outputs are likely not close due to wrong pointer arithmetic in the kernel.
    assert not torch.allclose(y, ref, atol=1e-3), "Kernel unexpectedly produced correct results on non-contiguous normalized dims."
