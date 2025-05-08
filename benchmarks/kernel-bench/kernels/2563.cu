#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: applies ReLU elementwise and simultaneously reduces the activated values
// using shared memory and warp-level primitives (__shfl_down_sync) for intra-block reduction.

template <typename scalar_t>
__global__ void fused_relu_reduction_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ blockSums,
    const int64_t size) {

    extern __shared__ scalar_t sdata[];  // shared memory to hold partial sums from each warp

    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    scalar_t thread_sum = static_cast<scalar_t>(0);

    // Grid-stride loop: compute ReLU and accumulate thread-local sum
    for (int i = globalIndex; i < size; i += stride) {
        scalar_t x = input[i];
        scalar_t y = x > static_cast<scalar_t>(0) ? x : static_cast<scalar_t>(0);
        output[i] = y;
        thread_sum += y;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's first thread writes its partial sum into shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from shared memory
    int numWarps = blockDim.x / warpSize;
    if (tid < numWarps) {
        scalar_t sum = sdata[tid];
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            blockSums[blockIdx.x] = sum;
        }
    }
}

// Reduction kernel to combine block-level sums into a final scalar
template <typename scalar_t>
__global__ void reduce_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               int n) {
    extern __shared__ scalar_t sdata[];
    int tid = threadIdx.x;

    scalar_t sum = static_cast<scalar_t>(0);
    if (tid < n)
        sum = input[tid];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}


// Forward function: fuses ReLU activation with reduction of the activated values.
// Returns a vector of two tensors: the elementwise ReLU output and a scalar tensor containing
// the sum of all ReLU outputs.

std::vector<torch::Tensor> forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Allocate temporary tensor to hold per-block reduction results
    auto blockSums = torch::empty({blocks}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_relu_reduction_kernel", ([&] {
        int shmem_size = (threads / warpSize) * sizeof(scalar_t);
        fused_relu_reduction_kernel<scalar_t><<<blocks, threads, shmem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            blockSums.data_ptr<scalar_t>(),
            size);
    }));

    // Allocate tensor for final reduction result
    auto final_sum = torch::empty({1}, input.options());
    
    if (blocks > 1) {
        // Determine threads for reduction kernel (power-of-two >= blocks)
        int reductionThreads = 1;
        while (reductionThreads < blocks) {
            reductionThreads *= 2;
        }
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_kernel", ([&] {
            reduce_kernel<scalar_t><<<1, reductionThreads, reductionThreads * sizeof(scalar_t)>>>(
                blockSums.data_ptr<scalar_t>(),
                final_sum.data_ptr<scalar_t>(),
                blocks);
        }));
    } else {
        final_sum.copy_(blockSums);
    }

    // Return both the elementwise ReLU output and the reduction sum
    return {output, final_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ReLU and Reduction forward (CUDA)");
}
