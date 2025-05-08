#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Templated kernel that uses a compile-time block size.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mse_forward_kernel_experiment(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    int tid = threadIdx.x;
    // Compute global index and stride based on BLOCK_SIZE
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    int stride = BLOCK_SIZE * gridDim.x;

    double local_sum = 0.0;
    // Grid-stride loop for processing elements
    // Process multiple elements per thread for better memory coalescing and instruction-level parallelism
    double diff;
    #pragma unroll 4
    for (int idx = gid; idx < num_elements; idx += stride) {
        diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        local_sum += diff * diff;
    }

    shm[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shm[tid] += shm[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

// Host function with an optional block_size parameter for experimentation
// Valid block sizes: 32, 64, 128, 256, 512. Defaults to 256 if unspecified or invalid.
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets, int block_size = 256) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Retrieve device properties to set grid size
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);

    // Launch a multiple of SM count for good occupancy (4 blocks per SM)
    int grid_size = sm_count * 4;

    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", ([&](){
        if (block_size == 32) {
            mse_forward_kernel_experiment<scalar_t, 32><<<grid_size, 32>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        } else if (block_size == 64) {
            mse_forward_kernel_experiment<scalar_t, 64><<<grid_size, 64>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        } else if (block_size == 128) {
            mse_forward_kernel_experiment<scalar_t, 128><<<grid_size, 128>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        } else if (block_size == 256) {
            mse_forward_kernel_experiment<scalar_t, 256><<<grid_size, 256>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        } else if (block_size == 512) {
            mse_forward_kernel_experiment<scalar_t, 512><<<grid_size, 512>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        } else {
            // Fallback to 256 if an unsupported block size is provided
            mse_forward_kernel_experiment<scalar_t, 256><<<grid_size, 256>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        }
    }));

    auto result = accumulator.div(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) with block size experiment",
          pybind11::arg("predictions"), pybind11::arg("targets"), pybind11::arg("block_size") = 256);
}
