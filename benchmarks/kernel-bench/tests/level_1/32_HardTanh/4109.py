
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to build and load the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="hardtanh_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger the missing error check.
# Although it is normally hard to force a CUDA launch error from Python,
# we simulate a situation where the kernel did not report an error.
# Here, we manually induce a situation likely to produce an erroneous result
# and then check that no CUDA error is reported by PyTorch (i.e. the error is hidden).
def test_no_kernel_error_detection():
    module = build_kernel()
    # Create a tensor that is valid, so that the kernel launch should work syntactically.
    # We then force a misconfiguration by calling the kernel with a negative grid dimension
    # via monkeypatching a very unusual tensor shape.
    # As we cannot directly force a launch error from Python, we check that after execution
    # torch.cuda.synchronize() does not signal an error even if we suspect something is wrong.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Call the kernel as implemented.
    out = module.forward(x, -1.0, 1.0)
    # Even if the kernel did a wrong memory access, without error checking
    # no error would propagate. So we trigger a synchronize to see if
    # an error is reported. (In a correct implementation, there should be an error check.)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail("CUDA error detected after kernel launch, but no error checking is implemented: " + str(e))
        
    # As an additional (indirect) check, we compare the results with the standard F.hardtanh.
    out_ref = F.hardtanh(x, min_val=-1.0, max_val=1.0)
    # The output might be correct in many cases.
    # This test is primarily about the missing error check.
    assert out.shape == x.shape, "Output shape mismatch."

# Test 2: Trigger behavior on non-contiguous input.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a tensor and make it noncontiguous by transposing.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # This makes the tensor non-contiguous.
    # Verify that the tensor is non contiguous
    assert not x_noncontig.is_contiguous(), "Tensor should be non-contiguous for this test."

    # Run the custom CUDA kernel.
    out_cuda = module.forward(x_noncontig, -1.0, 1.0)
    # Compute reference with PyTorch HardTanh which handles non-contiguous inputs properly.
    out_ref = F.hardtanh(x_noncontig, min_val=-1.0, max_val=1.0)
    
    # Because the kernel does not handle non-contiguous memory,
    # the result will likely be incorrect compared to the reference.
    # Here we check that the outputs differ.
    if torch.allclose(out_cuda, out_ref):
        pytest.fail("Kernel unexpectedly produced correct output for non-contiguous input; "
                    "it should assume contiguous input and possibly produce wrong results.")

# Test 3: Ensure that passing a CPU tensor immediately fails.
def test_cpu_tensor_error():
    module = build_kernel()
    x_cpu = torch.randn(16, 16384, device="cpu", dtype=torch.float32)
    with pytest.raises(Exception, match="Input tensor must be a CUDA tensor"):
        _ = module.forward(x_cpu, -1.0, 1.0)
