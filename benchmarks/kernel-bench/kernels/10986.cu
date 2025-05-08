#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void mse_forward_kernel_min_sync(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Each thread maintains its own accumulator
    double thread_sum = 0.0;
    
    // Process multiple elements per thread using grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_elements; 
         idx += blockDim.x * gridDim.x) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    // Warp-level reduction first (no sync needed within a warp)
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Only the first thread in each warp writes to shared memory
    __shared__ double warp_sums[8];  // For 256 threads = 8 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    
    // Single sync point needed here for shared memory consistency
    __syncthreads();

    // Final reduction by first warp only
    if (threadIdx.x < 8) {
        double sum = warp_sums[threadIdx.x];
        
        // Warp-level reduction of final sums (no sync needed)
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xff, sum, offset);
        }

        if (threadIdx.x == 0) {
            atomicAdd(sum_out, sum);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    const int grid_size = std::min(1024, (int)((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", ([&] {
        mse_forward_kernel_min_sync<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA) with minimal synchronization");
}