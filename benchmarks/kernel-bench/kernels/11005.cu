#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int VECTOR_SIZE = 2;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x * VECTOR_SIZE;
    
    double thread_sum = 0.0;

    // Process elements in aligned vector chunks
    int idx = tid * VECTOR_SIZE;
    while (idx < num_elements) {
        // Vector load for aligned, coalesced access
        float2 pred_vec = reinterpret_cast<const float2*>(preds)[idx/VECTOR_SIZE];
        float2 tgt_vec = reinterpret_cast<const float2*>(tgts)[idx/VECTOR_SIZE];
        
        // Process both elements unconditionally to avoid divergence
        double diff0 = static_cast<double>(pred_vec.x) - static_cast<double>(tgt_vec.x);
        double diff1 = static_cast<double>(pred_vec.y) - static_cast<double>(tgt_vec.y);
        thread_sum += diff0 * diff0 + diff1 * diff1;

        idx += stride;
    }

    // Warp-level reduction with shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Atomic add with one per warp
    if (threadIdx.x % 32 == 0) {
        atomicAdd(sum_out, thread_sum);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(),
               "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(),
               "Inputs must be contiguous");
    TORCH_CHECK(predictions.numel() == targets.numel(),
               "Input sizes must match");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Align elements to vector size
    int grid_size = (num_elements + BLOCK_SIZE * VECTOR_SIZE - 1) / (BLOCK_SIZE * VECTOR_SIZE);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    auto result = accumulator.div_(num_elements).to(predictions.dtype());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (vectorized warp optimized");
}