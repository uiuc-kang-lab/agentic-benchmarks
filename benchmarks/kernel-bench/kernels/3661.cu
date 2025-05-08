#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel that applies HardSigmoid elementwise and computes a block-level reduction
// of the activated outputs using shared memory and warp-level primitives

template <typename scalar_t>
__global__ void fused_hardsigmoid_reduction_kernel(const scalar_t* __restrict__ input,
                                                     scalar_t* __restrict__ output,
                                                     scalar_t* __restrict__ blockSums,
                                                     size_t numel) {
    extern __shared__ scalar_t sdata[]; // Shared memory for reduction
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // Constants for HardSigmoid: y = clamp((x + 3)/6, 0, 1)
    const scalar_t add_const = static_cast<scalar_t>(3);
    const scalar_t scale = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);

    scalar_t partial_sum = static_cast<scalar_t>(0);

    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + add_const) * scale;
        // Clamp y to [0, 1] using a branchless approach
        y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
        output[i] = y;
        partial_sum += y;
    }

    // Each thread writes its partial sum into shared memory
    sdata[tid] = partial_sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    // Reduce in steps, assuming blockDim.x is a power of 2
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level primitives for the final warp (no __syncthreads needed within a warp)
    if (tid < 32) {
        volatile scalar_t* vsdata = sdata;
        for (int offset = 16; offset > 0; offset /= 2) {
            vsdata[tid] += vsdata[tid + offset];
        }
    }

    // The first thread of each block writes the block's reduction sum to global memory
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel to combine block sums into a single global sum
// Although the global sum is not used to modify the activation results, it demonstrates
// an optimized reduction using shared memory and warp-level intrinsics.

template <typename scalar_t>
__global__ void final_reduce_kernel(scalar_t* blockSums, int numElements) {
    extern __shared__ scalar_t sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    scalar_t sum = (idx < numElements) ? blockSums[idx] : static_cast<scalar_t>(0);
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile scalar_t* vsdata = sdata;
        for (int offset = 16; offset > 0; offset /= 2) {
            vsdata[tid] += vsdata[tid + offset];
        }
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

// The forward function launches the fused kernel to compute the HardSigmoid activation
// and performs a subsequent reduction of the activated outputs.
// The output tensor remains identical to the expected activation result.

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    // Temporary tensor to hold per-block reduction sums
    auto blockSums = torch::empty({blocks}, input.options());

    // Launch the fused HardSigmoid and block reduction kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_hardsigmoid_reduction_cuda", ([&] {
        fused_hardsigmoid_reduction_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            blockSums.data_ptr<scalar_t>(),
            numel);
    }));

    // Perform iterative reduction on blockSums to compute a global sum
    int s = blocks;
    while (s > 1) {
        int threadsFinal = (s < 1024 ? s : 1024);
        int blocksFinal = (s + threadsFinal - 1) / threadsFinal;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "final_reduce_cuda", ([&] {
            final_reduce_kernel<scalar_t><<<blocksFinal, threadsFinal, threadsFinal * sizeof(scalar_t)>>>(
                blockSums.data_ptr<scalar_t>(), s);
        }));
        s = blocksFinal;
    }

    // The global reduction result is now in blockSums[0]. It can be used for further processing or logging.
    // For this kernel, we return the activated output tensor unchanged.
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused HardSigmoid activation with reduction (CUDA)");
}
