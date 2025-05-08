/*
 * This CUDA kernel implements an efficient softmax forward pass by combining the templated block size approach
 * from Kernel 2 with the warp-level reduction using shuffle intrinsics from Kernel 1.
 * It computes the per-row max and sum in two passes using warp-level reductions, then normalizes the output.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>


// Templated efficient softmax kernel with tunable BLOCK_SIZE
// Uses warp-level reductions (via __shfl_down_sync) to compute row max and sum in a row-wise softmax

template <int BLOCK_SIZE>
__global__ void softmax_efficient_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int num_features) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int warp_size = 32;
    const int lane_id = tid % warp_size;
    const int warp_id = tid / warp_size;
    // Calculate number of warps required for BLOCK_SIZE threads
    const int num_warps = (BLOCK_SIZE + warp_size - 1) / warp_size;

    // Pointers for the current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Shared memory will hold two arrays: first for per-warp max and then for per-warp sum
    extern __shared__ float sdata[]; // size: 2 * num_warps floats

    // -------------------------------
    // Step 1: Compute row maximum
    // -------------------------------
    float thread_max = -INFINITY;
    // Each thread processes multiple elements with striding
    for (int i = tid; i < num_features; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    // Intra-warp reduction using shuffle
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    // Each warp writes its maximum (from lane 0) to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = thread_max;
    }
    __syncthreads();

    float row_max = 0.0f;
    // First warp reduces the per-warp max values stored in shared memory
    if (tid < warp_size) {
        float max_val = (tid < num_warps) ? sdata[tid] : -INFINITY;
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) {
            row_max = max_val;
        }
    }
    __syncthreads();
    // Broadcast the computed row max to all threads
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // -------------------------------
    // Step 2: Compute exponentials and partial sum
    // -------------------------------
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += BLOCK_SIZE) {
        float exp_val = __expf(x_row[i] - row_max);
        y_row[i] = exp_val; // store intermediate exp values
        thread_sum += exp_val;
    }
    // Intra-warp reduction for sum using shuffle
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    // Each warp writes its partial sum to shared memory (offset by num_warps)
    if (lane_id == 0) {
        sdata[warp_id + num_warps] = thread_sum;
    }
    __syncthreads();

    float row_sum = 0.0f;
    // First warp reduces the partial sums from each warp
    if (tid < warp_size) {
        float sum_val = (tid < num_warps) ? sdata[tid + num_warps] : 0.0f;
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) {
            row_sum = sum_val;
        }
    }
    __syncthreads();
    // Broadcast the row sum to all threads
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // -------------------------------
    // Step 3: Normalize the output
    // -------------------------------
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < num_features; i += BLOCK_SIZE) {
        y_row[i] *= inv_sum;
    }
}


// Host function to launch the efficient softmax kernel with a tunable block size
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features, int block_size) {
    dim3 grid_dim(batch_size);
    int num_warps = (block_size + 31) / 32;
    int shared_mem_size = sizeof(float) * num_warps * 2;  // space for max and sum arrays

    switch(block_size) {
        case 32: {
            dim3 block_dim(32);
            softmax_efficient_kernel<32><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 64: {
            dim3 block_dim(64);
            softmax_efficient_kernel<64><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 128: {
            dim3 block_dim(128);
            softmax_efficient_kernel<128><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 256: {
            dim3 block_dim(256);
            softmax_efficient_kernel<256><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        case 512: {
            dim3 block_dim(512);
            softmax_efficient_kernel<512><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
        default: {
            // fallback to 256 if provided block size is unsupported
            dim3 block_dim(256);
            softmax_efficient_kernel<256><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
            break;
        }
    }
}

// PyTorch binding: Forward function
// Allows specifying an optional block_size parameter (default 256)

torch::Tensor forward(torch::Tensor x, int block_size = 256) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);

    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features, block_size);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient Softmax forward (CUDA) with tunable block size", 
          py::arg("x"), py::arg("block_size") = 256);
}
