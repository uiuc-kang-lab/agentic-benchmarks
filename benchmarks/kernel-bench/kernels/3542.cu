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

// Kernel 1: Apply SELU activation elementwise and perform intra-block reduction
// to compute the sum of activated values using shared memory and warp-level primitives.

template <typename scalar_t>
__global__ void selu_activation_reduce(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         size_t numel,
                                         scalar_t* block_sums) {
    extern __shared__ scalar_t sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    scalar_t local_sum = static_cast<scalar_t>(0);

    // SELU constants
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Each thread processes multiple elements strided by gridDim * blockDim
    for (size_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        scalar_t x = input[i];
        scalar_t res = (x > static_cast<scalar_t>(0))
                             ? x
                             : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        res = lambda * res;
        output[i] = res;
        local_sum += res;
    }

    // Store the local sum in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for the final stage
    if (tid < 32) {
        volatile scalar_t sum_val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) {
            block_sums[blockIdx.x] = sum_val;
        }
    }
}

// Kernel 2: Final reduction across block sums
// This kernel reduces the array of per-block sums to a single value using shared memory
// and warp-level primitives.

template <typename scalar_t>
__global__ void selu_final_reduce(scalar_t* block_sums, int numBlocks) {
    extern __shared__ scalar_t sdata[];
    unsigned int tid = threadIdx.x;
    scalar_t sum = (tid < numBlocks) ? block_sums[tid] : static_cast<scalar_t>(0);
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s && tid + s < numBlocks) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile scalar_t sum_val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) {
            block_sums[0] = sum_val;
        }
    }
}

// Host function: Executes the activation kernel as well as the reduction kernels.
// It returns a tuple containing the SELU-activated tensor and the overall reduction sum
// (i.e. the sum of all activated elements).

std::tuple<torch::Tensor, torch::Tensor> selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    // Allocate temporary tensor to hold per-block sums
    auto blockSums = torch::empty({blocks}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_activation_reduce_cuda", ([&] {
        int shmem_size = threads * sizeof(scalar_t);
        selu_activation_reduce<scalar_t><<<blocks, threads, shmem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel,
            blockSums.data_ptr<scalar_t>()
        );
    }));

    // Determine the next power-of-2 for the final reduction kernel
    int threads_final = 1;
    while (threads_final < blocks) threads_final <<= 1;
    if (threads_final < 32) threads_final = 32;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_final_reduce_cuda", ([&] {
        int shmem_size_final = threads_final * sizeof(scalar_t);
        selu_final_reduce<scalar_t><<<1, threads_final, shmem_size_final>>>(
            blockSums.data_ptr<scalar_t>(),
            blocks
        );
    }));

    // Extract the final reduction result (sum of all activated values) as a 1-element tensor
    auto final_sum = blockSums.slice(0, 0, 1).clone();

    return std::make_tuple(output, final_sum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Reduction Optimization (CUDA)");
}
