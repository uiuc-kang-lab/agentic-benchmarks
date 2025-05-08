
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Non-contiguous tensor (alignment and stride issue)
def test_noncontiguous_tensor():
    # Create a contiguous tensor and then make it non-contiguous with a transpose
    x = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32).transpose(0, 1)
    # Use dimension 1 (which is the reduction dimension in the original layout)
    # The correct torch.argmin result will be computed based on the tensor’s actual layout.
    ref = torch.argmin(x, dim=1)
    
    my_module = build_kernel()
    # Call the CUDA kernel – since the kernel expects contiguous layout,
    # the result may differ from the reference.
    out = my_module.forward(x, 1)
    torch.cuda.synchronize()

    # We expect a difference when non-contiguous
    assert not torch.equal(out, ref), "Kernel produced correct result on non-contiguous tensor unexpectedly."

# Test case 2: Tensor with non-standard strides
def test_nonstandard_strides():
    # Create a contiguous tensor and then slice it in a way that produces non-standard strides.
    base = torch.randn(10, 20, 30, device="cuda", dtype=torch.float32)
    x = base[:, ::2, :]  # slicing makes a non-contiguous tensor.
    ref = torch.argmin(x, dim=1)
    
    my_module = build_kernel()
    out = my_module.forward(x, 1)
    torch.cuda.synchronize()

    # We expect that the kernel (which uses raw pointer arithmetic assuming contiguity) will produce wrong results.
    assert not torch.equal(out, ref), "Kernel produced correct result on non-standard stride tensor unexpectedly."

# Test case 3: Half precision comparison issues
def test_half_precision_comparison():
    # Create a tensor of type half
    x = torch.randn(8, 16, device="cuda", dtype=torch.half)
    ref = torch.argmin(x, dim=1)
    
    my_module = build_kernel()
    out = my_module.forward(x, 1)
    torch.cuda.synchronize()

    # Due to potential improper __half comparisons in the kernel, the results may not match.
    assert not torch.equal(out, ref), "Kernel produced correct result on half precision tensor unexpectedly."

# Test case 4: Empty reduction dimension (K == 0)
def test_empty_reduction_dimension():
    # Create a tensor where the reduction dimension has size 0.
    x = torch.empty(5, 0, 7, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel does not check for K == 0 so reading from x may crash or throw an error.
        _ = my_module.forward(x, 1)
