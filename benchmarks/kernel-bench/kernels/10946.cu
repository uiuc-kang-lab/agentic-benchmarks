#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int VEC_SIZE = 4;

template <typename scalar_t>
__global__ void mse_vectorized_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double smem[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int vec_idx = (blockIdx.x * blockDim.x + tid) * VEC_SIZE;
    double thread_sum = 0.0;

    // Vectorized grid-stride loop
    for (int i = vec_idx; i < num_elements; i += blockDim.x * gridDim.x * VEC_SIZE) {
        scalar_t pred_vec[VEC_SIZE];
        scalar_t tgt_vec[VEC_SIZE];
        
        // Load vectorized elements
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            if (i + v < num_elements) {
                pred_vec[v] = preds[i + v];
                tgt_vec[v] = tgts[i + v];
            }
        }

        // Compute squared differences
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            if (i + v < num_elements) {
                double diff = static_cast<double>(pred_vec[v]) - static_cast<double>(tgt_vec[v]);
                thread_sum += diff * diff;
            }
        }
    }

    // Warp-level reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    // First lane in warp stores to shared memory
    if (tid % warpSize == 0)
        smem[tid / warpSize] = thread_sum;
    __syncthreads();

    // Final block reduction
    if (tid < warpSize) {
        double block_sum = (tid < blockDim.x / warpSize) ? smem[tid] : 0.0;
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        
        if (tid == 0)
            atomicAdd(sum_out, block_sum);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Reduced grid size due to vectorization
    int grid_size = (num_elements + (BLOCK_SIZE * VEC_SIZE) - 1) / (BLOCK_SIZE * VEC_SIZE);
    grid_size = std::min(grid_size, 512);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_vectorized_cuda", ([&] {
        mse_vectorized_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Vectorized MSE with Minimal Atomics (CUDA)");
}