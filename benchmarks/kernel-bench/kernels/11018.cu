#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 1024;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;

    // Grid-stride loop with uniform memory access
    for(int idx = global_idx; idx < num_elements; idx += gridDim.x * blockDim.x) {
        const double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    // Warp-level reduction first
    for(int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Store warp sums in shared memory
    if(threadIdx.x % 32 == 0) {
        shm[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    // Final block reduction
    if(threadIdx.x < BLOCK_SIZE/32) {
        double warp_sum = shm[threadIdx.x];
        for(int offset = BLOCK_SIZE/64; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if(threadIdx.x == 0) {
            atomicAdd(sum_out, warp_sum);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "Input sizes must match");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Calculate grid size to fill all SMs
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int grid_size = num_sms * 4;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    return accumulator.div_(num_elements).to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA optimized warp execution");
}