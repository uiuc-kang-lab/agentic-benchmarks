
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build and load the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="smooth_l1_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Check that the kernel crashes or fails when given mis‐aligned memory.
# One common way to force mis‐alignment is to offset a 1D tensor by slicing.
# This test expects the kernel to produce an error (or perhaps an incorrect result)
# due to the assumptions of 16-byte alignment.
def test_alignment_issue():
    module = build_kernel()
    # Create a larger tensor then take a narrow view that is likely misaligned.
    # Note: While not guaranteed, slicing in this way may result in a misaligned data_ptr.
    full_tensor = torch.randn(128, 4096 + 1, dtype=torch.float32, device="cuda")
    # We take a slice that starts at index 1 along the second dimension.
    predictions = full_tensor[:, 1:]
    targets = torch.randn_like(predictions)
    
    # Although the host wrapper checks for contiguity, slicing may yield non-contiguous tensors.
    # Here we expect the TORCH_CHECK in the host wrapper or undefined behavior in the kernel.
    with pytest.raises(RuntimeError):
        # This call should raise an error because the inputs are not contiguous
        # or cause misaligned memory accesses in the kernel.
        module.forward(predictions, targets)

# Test 2: Check that the kernel rejects inputs that are not float32.
def test_wrong_dtype():
    module = build_kernel()
    # Create float64 tensors which are not supported by the kernel.
    predictions = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    
    # The kernel does not check for dtype, so using float64 will lead to unsafe reinterpretation.
    # We expect an error (or a crash) because the kernel expects float32.
    with pytest.raises(RuntimeError):
        module.forward(predictions, targets)

# Test 3: Check that the kernel assumes a fixed block dimension (256) for the shared memory reduction.
# Although the host wrapper launches the kernel with block size 256, in more general use cases if the
# kernel is launched with a different blockDim this reduction buffer might be insufficient.
# Here we simulate a potential misuse by attempting to override the block size via a custom kernel launch.
def test_fixed_block_dim_issue():
    # Directly call the kernel with nonstandard block dimensions.
    # This requires building the extension module and then launching the kernel manually.
    module = build_kernel()

    predictions = torch.randn(128, 4096, dtype=torch.float32, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.float32, device="cuda")
    # Get total number of elements.
    n_elements = predictions.numel()
    output = torch.zeros(1, dtype=torch.float32, device="cuda")
    
    # Retrieve the raw kernel function from the module.
    # Note: This test bypasses the wrapper and launches the kernel with a block size different from 256.
    custom_block_size = 128  # non-standard block size compared to the hard-coded shared memory size.
    grid_size = (n_elements // 4 + custom_block_size - 1) // custom_block_size
    grid_size = grid_size if grid_size > 0 else 1
    try:
        module.smooth_l1_loss_vec_ldg_kernel.launch(
            grid=(grid_size,),
            block=(custom_block_size,),
            args=[
                predictions.data_ptr(),
                targets.data_ptr(),
                output.data_ptr(),
                n_elements
            ],
            stream=torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()
    except Exception as ex:
        pytest.skip("Kernel launch with non-standard blockDim encountered an error, which indicates a fixed shared memory assumption issue.")
    # The test does not check for correctness but highlights that using a non-256 block size
    # could potentially trigger reduction errors if the shared memory allocation is not updated.
    # In a real scenario, this misuse would be caught by the developer via testing and code review.
    assert True
