#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <tuple>

// Device helper: inline exponential function for float and double
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// Warp-level reduction using __shfl_down_sync
template <typename scalar_t>
__inline__ __device__ scalar_t warpReduceSum(scalar_t val) {
    // Use full warp mask
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel that applies SELU activation elementwise and concurrently reduces the sum of outputs.
// Each thread computes the SELU activation, then we perform an intra-block reduction
// using shared memory and warp-level primitives. The block sum is accumulated atomically
// into a global sum variable.

template <typename scalar_t>
__global__ void selu_kernel_reduction(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        scalar_t* __restrict__ global_sum,
                                        size_t numel) {
    extern __shared__ char smem[]; // shared memory allocated as raw bytes
    // reinterpret shared memory as an array of scalar_t
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t val = static_cast<scalar_t>(0);

    if (idx < numel) {
        scalar_t x = input[idx];
        // Compute SELU: lambda * (x if x > 0 else alpha * (exp(x)-1))
        scalar_t activated = (x > static_cast<scalar_t>(0))
                              ? x
                              : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        activated = static_cast<scalar_t>(1.05070098735548049342) * activated;
        output[idx] = activated;
        val = activated;
    }

    // Intra-warp reduction using shuffle
    unsigned int lane = threadIdx.x % warpSize;
    val = warpReduceSum(val);

    // Write each warp's sum to shared memory (only lane 0 of each warp does so)
    if(lane == 0) {
        int warpId = threadIdx.x / warpSize;
        sdata[warpId] = val;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp sums stored in shared memory
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    scalar_t block_sum = static_cast<scalar_t>(0);
    if(threadIdx.x < numWarps) {
        block_sum = sdata[threadIdx.x];
        block_sum = warpReduceSum(block_sum);
    }

    // Thread 0 of the block atomically adds the block's sum to the global sum
    if(threadIdx.x == 0) {
        atomicAdd(global_sum, block_sum);
    }
}

// Host function exposed via Pybind11. It launches the SELU kernel with integrated reduction.
// The function returns a tuple: the output tensor with the SELU activated values and a 1-element tensor
// containing the sum of all activated values. This fused approach reduces runtime by avoiding a separate
// reduction kernel and optimizing intra-block reductions with shared memory and warp-level primitives.

std::tuple<torch::Tensor, torch::Tensor> selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    auto global_sum_tensor = torch::zeros({1}, input.options());
    const size_t numel = input.numel();

    // Choosing 1024 threads per block for high occupancy
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    
    // Calculate shared memory size needed: one scalar per warp
    const int warpsPerBlock = (threads + 31) / 32;
    const size_t sharedMemSize = warpsPerBlock * sizeof(scalar_t);
    // Instead of the above tricky computation, we use a template-friendly approach in the dispatch below.

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const auto* input_ptr = input.data_ptr<scalar_t>();
        auto* output_ptr = output.data_ptr<scalar_t>();
        auto* global_sum_ptr = global_sum_tensor.data_ptr<scalar_t>();
        selu_kernel_reduction<scalar_t><<<blocks, threads, warpsPerBlock * sizeof(scalar_t)>>>(
            input_ptr,
            output_ptr,
            global_sum_ptr,
            numel
        );
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, global_sum_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Reduction (CUDA)");
}
