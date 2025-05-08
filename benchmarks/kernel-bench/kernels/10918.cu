#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

// First kernel: Compute per-block partial sum of squared differences
// Each block reduces its assigned elements in shared memory and writes its partial sum

template <typename scalar_t>
__global__ void mse_partial_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial,
    const int64_t num_elements
) {
    __shared__ double sdata[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double sum = 0.0;
    
    // Grid-stride loop: each thread processes multiple elements
    for (; idx < num_elements; idx += stride) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        sum += diff * diff;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // In-block reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write the block's result to the partial sum array
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Second kernel: Hierarchically reduce an array of doubles to a smaller array
// This kernel reduces the input array (of length n) by a factor of ~2*BLOCK_SIZE

__global__ void reduce_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int n
) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    double sum = 0.0;
    
    if (i < n) {
        sum = input[i];
        if (i + blockDim.x < n)
            sum += input[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


// Forward function performing two-stage reduction for MSE computation

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Determine grid size for the first kernel
    int numBlocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate tensor for partial sums in double precision
    auto partial = torch::empty({numBlocks}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_partial_cuda", ([&] {
        mse_partial_kernel<scalar_t><<<numBlocks, BLOCK_SIZE, 0, stream>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            partial.data_ptr<double>(),
            num_elements
        );
    }));

    // Iteratively reduce the partial sums array without using global atomics
    int current_size = numBlocks;
    while (current_size > 1) {
        int threads = BLOCK_SIZE;
        int blocks = (current_size + threads * 2 - 1) / (threads * 2);
        auto temp = torch::empty({blocks}, predictions.options().dtype(at::kDouble));
        size_t shared_mem = threads * sizeof(double);

        reduce_kernel<<<blocks, threads, shared_mem, stream>>>(
            partial.data_ptr<double>(),
            temp.data_ptr<double>(),
            current_size
        );
        partial = temp;
        current_size = blocks;
    }

    // The final sum is stored in partial[0]
    auto total_sum = partial.item<double>();
    double mse = total_sum / static_cast<double>(num_elements);

    // Create the result tensor and convert it to the original type
    auto result = torch::full({1}, mse, predictions.options().dtype(at::kDouble));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) with two-stage reduction");
}
