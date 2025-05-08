#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <tuple>

// GELU function specializations for float and double

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x);

template <>
__device__ inline float gelu_function<float>(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

// Fused kernel: Computes GELU activation element-wise and performs an intra-block reduction
// using shared memory and warp-level primitives to sum the GELU outputs per block.

template <typename scalar_t>
__global__ void gelu_fused_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ y,
                                  scalar_t* __restrict__ block_sums,
                                  size_t numel) {
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    scalar_t partial_sum = 0;

    // Grid-stride loop: compute GELU for each element and accumulate partial sum
    for (size_t i = global_tid; i < numel; i += stride) {
        scalar_t val = gelu_function<scalar_t>(x[i]);
        y[i] = val;
        partial_sum += val;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Each warp's lane 0 writes its reduced sum into shared memory
    __shared__ scalar_t shared[32];  // Max number of warps per block (assuming blockDim.x <= 1024)
    int warp_id = tid / warpSize;
    if ((tid & (warpSize - 1)) == 0) {
        shared[warp_id] = partial_sum;
    }
    __syncthreads();

    // First warp reduces the sums from shared memory
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < num_warps) {
        scalar_t block_sum = shared[tid];
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (tid == 0) {
            block_sums[blockIdx.x] = block_sum;
        }
    }
}

// Final reduction kernel: Reduces an array of block sums to a single global sum

template <typename scalar_t>
__global__ void final_reduce_kernel(const scalar_t* __restrict__ in,
                                      scalar_t* __restrict__ out,
                                      size_t n) {
    scalar_t sum = 0;
    int tid = threadIdx.x;
    
    // Sum over the input array in a grid-stride manner
    for (size_t i = tid; i < n; i += blockDim.x) {
        sum += in[i];
    }
    
    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Use shared memory to reduce across warps
    __shared__ scalar_t shared[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (tid < blockDim.x / warpSize) {
        sum = shared[tid];
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (tid == 0) {
            out[0] = sum;
        }
    }
}

// Forward function: Computes the GELU activation and also returns the global sum of the GELU outputs
// as a demonstration of optimized reduction using shared memory and warp-level primitives.

std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    // Allocate tensor for block sums (one per block)
    auto block_sums = torch::empty({blocks}, options);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_fused_cuda", ([&] {
        gelu_fused_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            numel);
    }));

    // Allocate a tensor to hold the final global sum
    auto global_sum = torch::empty({1}, options);
    
    // Launch the final reduction kernel with 1 block
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "final_reduce_cuda", ([&] {
        final_reduce_kernel<scalar_t><<<1, threads>>>(
            block_sums.data_ptr<scalar_t>(),
            global_sum.data_ptr<scalar_t>(),
            blocks);
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    
    return std::make_tuple(output, global_sum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward with fused reduction (CUDA)");
}
