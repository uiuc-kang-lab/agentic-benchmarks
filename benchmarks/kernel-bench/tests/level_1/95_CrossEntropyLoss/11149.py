
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper: compile and load the CUDA extension from kernel.cu
def build_kernel():
    # Make sure the file kernel.cu exists in the current directory.
    module = load(
        name="cross_entropy_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Noncontiguous predictions tensor.
def test_noncontiguous_predictions():
    cuda_mod = build_kernel()
    batch_size = 4096
    num_classes = 10
    # Create a contiguous tensor and then apply a transpose (or slice) to make it noncontiguous.
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # For example, using .t() (transpose) will change the storage order.
    noncontig_predictions = predictions.t()  # shape [num_classes, batch_size], non-contiguous in desired layout
    # Transpose back but without making it contiguous (just view it differently)
    noncontig_predictions = noncontig_predictions.t()  # now shape is same as original but may not be contiguous
    assert not noncontig_predictions.is_contiguous(), "Predictions tensor should be noncontiguous for this test."
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    with pytest.raises(RuntimeError):
        # The kernel’s pointer arithmetic is based on contiguous layout. Even if the kernel does not
        # explicitly check contiguity, the results or memory access may be incorrect.
        loss = cuda_mod.forward(noncontig_predictions, targets)
        torch.cuda.synchronize()

# Issue 2: Invalid target indices (out-of-bound targets).
def test_invalid_target_indices():
    cuda_mod = build_kernel()
    batch_size = 4096
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Create targets where one target is set equal to num_classes (i.e. out-of-bound)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    targets[0] = num_classes  # invalid index!
    with pytest.raises(RuntimeError):
        # We expect an invalid memory access or assertion inside the kernel.
        loss = cuda_mod.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 3: Kernel reduction relies on blockDim.x being a power-of-2.
# Although the forward() always launches with 512 threads (which is a power-of-2),
# we simulate a scenario where an alternative launch configuration is used.
# To test this we recompile the kernel with a modified launch configuration.
def test_non_power_of_two_threads(tmp_path):
    # Create a temporary kernel file that forces a non-power-of-2 thread count.
    kernel_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void cross_entropy_loss_kernel_2d(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x; 

    extern __shared__ float sdata[];
    float* smax = sdata;
    float* ssum = sdata + blockSize;

    const float* sample_logits = logits + sample_idx * num_classes;

    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockSize) {
        float val = sample_logits[j];
        if (val > local_max) local_max = val;
    }
    smax[tid] = local_max;
    __syncthreads();

    // Reduction for maximum (assumes blockSize is a power-of-2)
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = smax[0];

    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockSize) {
        local_sum += expf(sample_logits[j] - global_max);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = ssum[0];

    if (tid == 0) {
        int target = targets[sample_idx];
        float loss = -(sample_logits[target] - global_max - logf(global_sum));
        losses[sample_idx] = loss;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch one block per sample, but using a non-power-of-2 thread count, e.g., 500.
    const int threads = 500;
    const int blocks = batch_size;
    size_t shared_mem_size = 2 * threads * sizeof(float);

    cross_entropy_loss_kernel_2d<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in cross_entropy_loss_kernel_2d: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with non-power-of-two threads (CUDA)");
}
'''
    # Write the modified kernel into a temporary file.
    kernel_file = tmp_path / "non_power_two_kernel.cu"
    kernel_file.write_text(kernel_source)
    
    non_power_mod = load(
        name="non_power_two_kernel",
        sources=[str(kernel_file)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    
    batch_size = 128
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    # Since the reduction algorithm may fail with a non-power-of-two thread block,
    # we anticipate that the kernel will produce an incorrect result.
    # We compare against the correct loss computed using PyTorch.
    loss_kernel = non_power_mod.forward(predictions, targets)
    loss_ref = torch.nn.functional.cross_entropy(predictions, targets)
    
    # The absolute difference likely is large when the reduction does not work.
    # We trigger an assertion if the difference is below a small threshold (meaning the bug did not surface).
    diff = (loss_kernel - loss_ref).abs().item()
    assert diff > 1e-3, f"Kernel reduction with non-power-of-two threads did not trigger an error, diff = {diff}"

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
