#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int UNROLL_FACTOR = 4;  // Unroll factor for loop unrolling

// Kernel with loop unrolling using #pragma unroll
template <typename scalar_t>
__global__ void mse_forward_kernel_unroll_pragma(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    double local_sum = 0.0;

    // Unrolled loop for processing elements
    #pragma unroll UNROLL_FACTOR
    for (int idx = gid; idx < num_elements; idx += stride * UNROLL_FACTOR) {
        double diff[UNROLL_FACTOR];
        
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            int unrolled_idx = idx + i * stride;
            if (unrolled_idx < num_elements) {
                diff[i] = static_cast<double>(preds[unrolled_idx]) - static_cast<double>(tgts[unrolled_idx]);
                local_sum += diff[i] * diff[i];
            }
        }
    }

    shm[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm[tid] += shm[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel_unroll_pragma<scalar_t><<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) with loop unrolling");
}