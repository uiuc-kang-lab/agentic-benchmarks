#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int WARP_SIZE = 32;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE/WARP_SIZE];
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    int idx = blockIdx.x * blockDim.x + tid;
    
    double thread_sum = 0.0;
    
    // Strided loop with minimal divergence
    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += blockDim.x * gridDim.x;
    }
    
    // Warp-level reduction using shuffle
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shm[wid] = thread_sum;
    }
    __syncthreads()
    
    // Final reduction across warps
    if (tid < (BLOCK_SIZE/WARP_SIZE)) {
        double warp_sum = shm[tid];
        if (tid == 0) {
            for (int i = 1; i < (BLOCK_SIZE/WARP_SIZE); ++i) {
                warp_sum += shm[i];
            }
            atomicAdd(sum_out, warp_sum);
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

    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}