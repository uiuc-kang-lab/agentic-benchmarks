#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <vector>

namespace cg = cooperative_groups;

// Fused kernel: Computes the Frobenius norm (sum of squares reduction) and normalizes the tensor
// in a single kernel launch using Cooperative Groups for grid-wide synchronization.
// Note: This kernel requires a CUDA device and driver that supports cooperative kernel launches.
__global__ void fused_norm_normalize_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             float* __restrict__ block_sums,
                                             float* __restrict__ norm_out,
                                             int numel) {
    // Obtain the grid group for grid-wide synchronization
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int global_idx = blockIdx.x * blockSize + tid;

    // Phase 1: Each thread computes its partial sum over elements using striding
    float sum_val = 0.0f;
    for (int i = global_idx; i < numel; i += gridDim.x * blockSize) {
        float val = input[i];
        sum_val += val * val;
    }

    // Use shared memory for in-block reduction
    __shared__ float sdata[256];
    sdata[tid] = sum_val;
    __syncthreads();

    // Standard tree reduction in shared memory
    for (int stride = blockSize / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    // Warp-level reduction without __syncthreads using volatile memory
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }

    // Write the block's partial sum to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }

    // Synchronize across the entire grid to ensure all block sums are written
    grid.sync();

    // Phase 2: Block 0 aggregates all block partial sums to compute the global sum
    if (blockIdx.x == 0) {
        __shared__ float s_global[256];  // Assumes blockDim.x is 256
        float partial = 0.0f;
        // Each thread in block 0 sums over a subset of block_sums
        for (int i = tid; i < gridDim.x; i += blockSize) {
            partial += block_sums[i];
        }
        s_global[tid] = partial;
        __syncthreads();
        
        // Reduce the partial sums in shared memory
        for (int stride = blockSize / 2; stride > 32; stride /= 2) {
            if (tid < stride) {
                s_global[tid] += s_global[tid + stride];
            }
            __syncthreads();
        }
        if (tid < 32) {
            volatile float* vs_global = s_global;
            vs_global[tid] += vs_global[tid + 32];
            vs_global[tid] += vs_global[tid + 16];
            vs_global[tid] += vs_global[tid + 8];
            vs_global[tid] += vs_global[tid + 4];
            vs_global[tid] += vs_global[tid + 2];
            vs_global[tid] += vs_global[tid + 1];
        }
        if (tid == 0) {
            // Compute the Frobenius norm
            norm_out[0] = sqrt(s_global[0]);
        }
    }

    // Synchronize so that every block gets the computed norm
    grid.sync();

    // Phase 3: Normalize the tensor using the computed norm
    float norm_val = norm_out[0];
    for (int i = global_idx; i < numel; i += gridDim.x * blockSize) {
        output[i] = input[i] / norm_val;
    }
}

// Host function interfacing with PyTorch. It allocates temporary tensors for block sums and the norm,
// then launches the fused kernel. The use of a single kernel eliminates extra kernel launch and host-device
// synchronization to retrieve the norm, improving efficiency.

torch::Tensor fused_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int numel = input.numel();
    const int threads = 256;
    const int blocks = std::min(65535, (numel + threads - 1) / threads);

    // Temporary tensor to hold partial block sums
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto block_sums_tensor = torch::zeros({blocks}, options);

    // Tensor to hold the computed norm (scalar)
    auto norm_tensor = torch::zeros({1}, options);

    // Launch the fused kernel. Note: This kernel uses cooperative groups and must be launched as a
    // cooperative kernel. Ensure your GPU supports cooperative kernel launches (CUDA 9+ and Volta+ recommended).
    fused_norm_normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        block_sums_tensor.data_ptr<float>(),
        norm_tensor.data_ptr<float>(),
        numel
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_forward, "Frobenius norm normalization using a fused kernel with cooperative groups");
}
