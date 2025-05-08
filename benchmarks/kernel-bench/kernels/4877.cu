#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using __shfl_down_sync for final stages
__inline__ __device__ float warpReduceSum(float val) {
    // Reduce within a warp using shuffle instructions
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// Optimized L1 normalization kernel using shared memory for intra-block reductions and warp-level primitives for final reduction stages
__global__ void l1_norm_forward_kernel_warp_optimized(const float* __restrict__ x,
                                                       float* __restrict__ out,
                                                       int N,
                                                       int D) {
    extern __shared__ float sdata[];  // Shared memory for storing warp-level partial sums

    int row = blockIdx.x;          // Each block is responsible for one row
    int tid = threadIdx.x;         // Thread index within the block

    // Each thread computes a partial sum over its portion of the row
    float thread_sum = 0.0f;
    for (int col = tid; col < D; col += blockDim.x) {
        thread_sum += fabsf(x[row * D + col]);
    }

    // Use warp-level reduction via shfl_down_sync
    int lane = tid & 31;           // Lane index in the warp
    int warp_id = tid >> 5;        // Warp index in the block
    thread_sum = warpReduceSum(thread_sum);

    // Write the reduced sum of each warp to shared memory
    if (lane == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Let the first warp load the partial sums and perform final reduction
    if (tid < (blockDim.x + 31) / 32) {
        thread_sum = sdata[lane];
        thread_sum = warpReduceSum(thread_sum);
        if (tid == 0) {
            sdata[0] = thread_sum;  // sdata[0] now holds the total sum of the row
        }
    }
    __syncthreads();

    float total_sum = sdata[0];
    // Avoid division by zero
    if (total_sum == 0.0f) {
        total_sum = 1e-12f;
        if (tid == 0) {
            sdata[0] = total_sum;
        }
    }
    __syncthreads();

    // Normalize the elements of the row
    for (int col = tid; col < D; col += blockDim.x) {
        out[row * D + col] = x[row * D + col] / total_sum;
    }
}

// Host function exposed to PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    // Choose number of threads per block (capped at 1024 or D, whichever is smaller)
    int threads = std::min<int>(1024, D);
    // Compute shared memory size: one float per warp
    int num_warps = (threads + 31) / 32;
    int shared_mem_size = num_warps * sizeof(float);

    l1_norm_forward_kernel_warp_optimized<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA) with warp-level optimized reductions");
}
