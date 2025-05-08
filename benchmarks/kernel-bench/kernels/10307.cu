#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that computes GELU activation for each element and, in parallel,
// performs an intra-block reduction (summing the computed results) using shared memory
// and warp-level primitives (__shfl_down_sync). The reduction is computed per block
// and stored in a separate array. The final activation output is written to y, ensuring
// correctness compared to the original implementation.

__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, float* __restrict__ block_sums, int n) {
    // Allocate dynamic shared memory for reduction
    extern __shared__ float shared_data[];  // size = blockDim.x

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float result = 0.0f;
    if (idx < n) {
        float xi = x[idx];
        float x_cubed = xi * xi * xi;
        float inner = xi + 0.044715f * x_cubed;
        inner *= 0.7978845608f;  // sqrt(2/pi)
        float tanh_val = tanhf(inner);
        result = 0.5f * xi * (1.0f + tanh_val);
        y[idx] = result;
    } else {
        result = 0.0f;
    }

    // Each thread writes its computed result into shared memory
    shared_data[tid] = result;
    __syncthreads();

    // Use warp-level primitives to perform reduction of the block's results
    unsigned int mask = 0xffffffff;  // full warp mask
    float val = shared_data[tid];

    // Intra-warp reduction using __shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // Write the warp-level reduced result to shared memory
    int warp_id = tid / warpSize;
    if ((tid % warpSize) == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();

    // Final reduction across warps in the block
    if (tid < (blockDim.x / warpSize)) {
        float block_sum = shared_data[tid];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (tid == 0) {
            block_sums[blockIdx.x] = block_sum;
        }
    }
}

// Host function that launches the kernel
// Note: While the kernel computes an additional reduction (block_sums),
// the returned result is the tensor y (GELU activation) which matches the original behavior.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate an extra tensor to store per-block reduction sums (for demonstration/possible
    // future fusion with other reduction operations)
    auto block_sums = torch::empty({blocks}, x.options());

    size_t shared_mem_bytes = threads * sizeof(float);

    gelu_kernel<<<blocks, threads, shared_mem_bytes>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU forward CUDA implementation with reduction");
}
