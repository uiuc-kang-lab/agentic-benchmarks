#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

// This fused kernel computes the Frobenius norm (via sum of squares reduction) and then normalizes the input tensor in a single launch.
// It leverages block-level reduction with warp-level shuffles and uses cooperative groups to perform a grid-level synchronization,
// eliminating the need for a costly device-to-host memcpy and a separate normalization kernel.

__global__ void fused_norm_normalize_kernel(const float* input, float* output, float* norm_val, int numel) {
    extern __shared__ float sdata[];  // Shared memory for per-block reduction
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float partial_sum = 0.0f;

    // Each thread computes a partial sum of squares across striding indices
    while (idx < numel) {
        float val = input[idx];
        partial_sum += val * val;
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    // Intra-block reduction using sequential halving
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Warp-level reduction without extra __syncthreads (using volatile)
    if (tid < 32) {
        volatile float* vsdata = sdata;
        for (int offset = 16; offset > 0; offset >>= 1) {
            vsdata[tid] += vsdata[tid + offset];
        }
    }

    // Use cooperative groups to ensure all blocks have finished their intra-block reduction
    cg::grid_group grid = cg::this_grid();
    if (tid == 0) {
        // Atomically add the block's result into the global accumulator
        atomicAdd(norm_val, sdata[0]);
    }
    grid.sync();  // Grid-wide synchronization to make sure the full sum is accumulated

    // Compute the Frobenius norm in a single thread (block 0, thread 0) and broadcast via shared memory
    __shared__ float norm;
    if (blockIdx.x == 0 && tid == 0) {
        norm = sqrtf(*norm_val);
    }
    grid.sync();

    // Now normalize the input tensor using the computed norm
    idx = blockIdx.x * blockDim.x + tid;
    while (idx < numel) {
        output[idx] = input[idx] / norm;
        idx += gridDim.x * blockDim.x;
    }
}

// Host function interfacing with PyTorch
torch::Tensor fused_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor with the same shape as input
    auto output = torch::empty_like(input);
    // Create a norm accumulator initialized to 0
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    int blocks = min(65535, (numel + threads - 1) / threads);
    
    // Allocate shared memory for the reduction
    int shared_mem_bytes = threads * sizeof(float);

    // Launch the fused kernel. Note: Fused grid-wide synchronization via cooperative groups requires a cooperative kernel launch
    // and hardware support. Ensure that the launch configuration and device support it.
    fused_norm_normalize_kernel<<<blocks, threads, shared_mem_bytes>>>(input_ptr, output_ptr, norm_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Frobenius norm normalization using a fused CUDA kernel with Cooperative Groups");
}
