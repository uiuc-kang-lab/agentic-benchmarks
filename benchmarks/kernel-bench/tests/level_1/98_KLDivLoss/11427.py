
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the kernel module from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_incorrect_kl_div_formula_and_normalization():
    """
    Test case to trigger issues 1 and 2.
    We construct a trivial case where predictions=[1.0] (so log(1)=0) and targets=[1.0].
    According to torch.nn.functional.kl_div with reduction='batchmean', the expected loss is 0,
    because the correct formula is 1*(log(1) - log(1)) = 0.
    However, the kernel computes exp(0) - 1*0 = 1 and then divides by n (which is 1 here), yielding 1.
    Thus, the kernel result should differ from the expected value.
    """
    predictions = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    targets = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    # The kernel expects log_predictions, matching the PyTorch function call in Model.forward.
    log_predictions = torch.log(predictions)
    
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")
    # The expected value should be 0 while kernel_out is expected to be nonzero (1 in this trivial case).
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Kernel output ({kernel_out}) unexpectedly matches the expected KL divergence ({expected})."
    )

def test_input_tensor_not_float32():
    """
    Test case to trigger an issue related to input type.
    The kernel expects float32 tensors (as it directly casts the data pointer to float*).
    Here we supply float64 tensors to force a type incompatibility.
    We expect the kernel to raise a RuntimeError.
    """
    predictions = torch.tensor([[0.3, 0.7]], dtype=torch.float64, device="cuda").softmax(dim=-1)
    targets = torch.tensor([[0.3, 0.7]], dtype=torch.float64, device="cuda").softmax(dim=-1)
    log_predictions = torch.log(predictions)
    
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should fail because the kernel code expects float32 data.
        _ = kernel_module.forward(log_predictions, targets)
        torch.cuda.synchronize()

def test_non_multiple_of_warp_size():
    """
    Test case to trigger the warp-related assumption (issue 4).
    The kernel launch configuration is hardcoded to use 256 threads per block,
    which is assumed to be divisible by the warp size (32). If we supply an input
    whose total number of elements is significantly less than the number of threads,
    the reduction may involve threads that do not have corresponding valid indices,
    and the computation of the warp_results may be misaligned.
    
    Here we pick n (number of elements) less than blockDim (256) so that many threads do
    not participate in valid computation. We expect the kernel output to be incorrect.
    """
    # Choose n such that n < 256.
    n = 100  
    predictions = torch.ones((n,), dtype=torch.float32, device="cuda")
    targets = torch.ones((n,), dtype=torch.float32, device="cuda")
    log_predictions = torch.log(predictions)
    
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # For ones, since log(1)==0, the correct KL divergence should be 0.
    expected = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    # Expect that the kernel output is not close to the expected value (0) due to reduction issues.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Kernel output ({kernel_out}) unexpectedly equals expected value ({expected}), "
        "which suggests a warp reduction or block-size assumption issue."
    )

def test_min_macro_compilation():
    """
    Although this test runs at runtime, it indirectly verifies that the host code successfully compiled.
    It attempts to run the kernel normally. 
    If the use of min(256, ...) in the host code is not handled properly (e.g. missing namespace or header),
    then the module would not have compiled.
    We check that the module can be built and executed with a simple invocation.
    """
    predictions = torch.tensor([[0.2, 0.8]], dtype=torch.float32, device="cuda").softmax(dim=-1)
    targets = torch.tensor([[0.2, 0.8]], dtype=torch.float32, device="cuda").softmax(dim=-1)
    log_predictions = torch.log(predictions)
    
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    # No assertion here - successful build and execution means the min macro use didn't break compilation.
    assert kernel_out.numel() == 1, "Kernel output should be a scalar tensor."

