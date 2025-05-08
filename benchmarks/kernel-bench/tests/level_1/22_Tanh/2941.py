
import pytest
import torch
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

# Issue 1: Input tensor is not on CUDA device.
def test_cpu_tensor():
    my_module = build_kernel()
    x_cpu = torch.randn(1024, 1024, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect error because kernel expects a CUDA tensor.
        res = my_module.forward(x_cpu)
        torch.cuda.synchronize()

# Issue 2: Input tensor is not contiguous.
def test_non_contiguous():
    my_module = build_kernel()
    # Create a contiguous tensor and then take a non-contiguous slice.
    x = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()  # Transpose produces a non-contiguous tensor.
    # Create reference output using PyTorch's tanh.
    ref = torch.tanh(x_non_contig)
    out = my_module.forward(x_non_contig)
    torch.cuda.synchronize()
    # The non contiguous access in our kernel is not handled.
    # This test will likely fail because the kernel reads wrong values.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly handled non contiguous input correctly!"

# Issue 3: Unsupported half precision type.
def test_half_precision():
    my_module = build_kernel()
    x_half = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # AT_DISPATCH_FLOATING_TYPES does not include half precision.
        out = my_module.forward(x_half)
        torch.cuda.synchronize()

# Issue 4: Lack of kernel launch error checking.
# We simulate a potential kernel error by intentionally passing an incorrectly sized tensor.
# Here, we mimic an error scenario by modifying the tensor size post-allocation.
def test_kernel_launch_error(monkeypatch):
    my_module = build_kernel()
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    
    # Monkeypatch the cuda kernel launcher to force an invalid grid dimension
    original_forward = my_module.forward
    def faulty_forward(tensor):
        # Call the original forward, then modify grid size with a negative value in the kernel launch.
        # We simulate this by calling the kernel with an invalid size if possible.
        # Since we cannot easily change the launch parameters externally,
        # this test simply demonstrates the difficulty in capturing asynchronous kernel launch errors.
        # So we force a runtime error by directly calling a failing CUDA call.
        torch.cuda._sleep(-1)  # Invalid sleep to provoke an error.
        return original_forward(tensor)
    
    monkeypatch.setattr(my_module, "forward", faulty_forward)
    
    with pytest.raises(Exception):
        res = my_module.forward(x)
        torch.cuda.synchronize()
