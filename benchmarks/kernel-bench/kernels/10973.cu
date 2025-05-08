#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void mse_forward_kernel_uniform(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Each thread accumulates its own sum in double precision
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Compute uniform number of iterations for all threads
    int n_iter = (num_elements + stride - 1) / stride;

    // Loop uniformly for all threads with improved memory coalescing
    #pragma unroll 4
    for (int iter = 0; iter < n_iter; iter++) {
        int i = idx + iter * stride;
        if (i < num_elements) {  // Simplified boundary check
            double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
            thread_sum += diff * diff;
        }
    }

    // Warp-level reduction using cooperative groups
    thread_sum = cg::reduce(warp, thread_sum, cg::plus<double>());

    // Each warp's leader writes its reduced sum to shared memory
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    if (warp.thread_rank() == 0) {
        warp_sums[threadIdx.x / 32] = thread_sum;
    }
    block.sync();

    // First warp reduces the warp sums
    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        double block_sum = warp_sums[threadIdx.x];
        auto active_threads = cg::coalesced_threads();
        block_sum = cg::reduce(active_threads, block_sum, cg::plus<double>());
        
        if (active_threads.thread_rank() == 0) {
            atomicAdd(sum_out, block_sum);
        }
    }
}

// Host function launching the kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    // Use double precision accumulator to maintain numerical accuracy
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_uniform", ([&] {
        mse_forward_kernel_uniform<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final mean squared error: divide accumulated sum by number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA) with uniform iteration to minimize warp divergence");
}
