#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int WARP_SIZE = 32;
static const int ELEMENTS_PER_THREAD = 4;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[WARP_SIZE];
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp_count = BLOCK_SIZE / WARP_SIZE;
    
    // Calculate base index aligned to warp size
    unsigned int base_idx = (blockIdx.x * BLOCK_SIZE + tid) * ELEMENTS_PER_THREAD;
    const unsigned int stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    
    double thread_sum = 0.0;
    
    // Process multiple elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        unsigned int idx = base_idx + i;
        if (idx < num_elements) {
            double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
            thread_sum += diff * diff;
        }
        base_idx += stride;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shm[wid] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces results from all warps
    if (wid == 0) {
        thread_sum = (lane < warp_count) ? shm[lane] : 0.0;
        
        // Final warp-level reduction
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (lane == 0) {
            atomicAdd(sum_out, thread_sum);
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

    // Calculate grid size ensuring multiple of warp size
    const int grid_size = ((num_elements / ELEMENTS_PER_THREAD + BLOCK_SIZE - 1) / BLOCK_SIZE + WARP_SIZE - 1) & ~(WARP_SIZE - 1);

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