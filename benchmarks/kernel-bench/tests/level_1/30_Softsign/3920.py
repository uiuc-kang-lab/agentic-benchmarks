
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build/load our custom CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="softsign_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Lack of float32 type checking
# If we pass a tensor with dtype=torch.float64, the kernel will use x.data_ptr<float>()
# and interpret the data incorrectly. This test demonstrates that the output
# does not match what softsign should produce.
def test_non_float_input():
    my_module = build_kernel()
    # Create a contiguous double tensor (float64) on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Calculate expected result in double precision using the softsign formula.
    expected = x / (1 + torch.abs(x))
    # Pass the double tensor to our kernel extension.
    # Since CHECK_INPUT does not verify the dtype, the kernel will run with a wrong cast.
    out = my_module.forward(x)
    # Convert the kernel output to double for proper comparison.
    out_double = out.to(torch.float64)
    # The result should be different from the expected result.
    assert not torch.allclose(out_double, expected, atol=1e-5), (
        "Kernel incorrectly produced the expected result with a non-float32 tensor. "
        "This indicates it may be auto-casting or handling input types in an unintended way."
    )

# Issue 2: No error checking after kernel launch.
# Although it might be difficult to simulate a kernel launch error reliably,
# one can at least try to force the condition by triggering invalid memory access.
# We can generate a non-contiguous tensor to trigger the CHECK_CONTIGUOUS, which is not part
# of the kernel launch error check per se -- however, it shows that the input is not validated
# enough in terms of its memory layout.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then make a non-contiguous version via transpose.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_non_contig = x.t()  # Transpose makes it non contiguous.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        # This call is expected to throw due to the CHECK_CONTIGUOUS macro in our kernel.
        my_module.forward(x_non_contig)

# Issue 3: Misleading kernel name ("warp_optimized")
# There is no straightforward way in a unit test to detect that warp-level primitives
# are not used. However, we can include a placeholder test that alerts the developer about this.
@pytest.mark.skip(reason="Kernel naming issue: 'warp_optimized' is misleading. No functional test available.")
def test_kernel_name_misleading():
    # This test is skipped because the issue is semantic/documentation related.
    pass

# Issue 4: Hard-coded type (lack of templating/generalization)
# We already demonstrate in test_non_float_input that passing a non-float32 tensor yields wrong results.
# Therefore, we can treat that test as also illustrating Issue 4.
@pytest.mark.skip(reason="Issue already covered by test_non_float_input regarding support for non-float32 types.")
def test_kernel_general_dtype():
    pass
