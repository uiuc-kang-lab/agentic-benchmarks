#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>


// Define block size and maximum elements for constant memory
#define BLOCK_SIZE 256
// For float: 64KB / sizeof(float) = 16384 elements
#define MAX_CONST_FLOAT 16384
// For double: 64KB / sizeof(double) = 8192 elements
#define MAX_CONST_DOUBLE 8192

// Declare read-only target arrays in constant memory
__constant__ float const_tgts_f[MAX_CONST_FLOAT];
__constant__ double const_tgts_d[MAX_CONST_DOUBLE];

// Original kernel using global memory (fallback)
template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;

    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += blockDim.x * gridDim.x;
    }

    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

// New kernel using constant memory for targets
template <typename scalar_t>
__global__ void mse_forward_const_kernel(
    const scalar_t* __restrict__ preds,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;

    while (idx < num_elements) {
        double diff;
        if constexpr (std::is_same<scalar_t, float>::value) {
            // Read target from constant memory for float
            float t_val = const_tgts_f[idx];
            diff = static_cast<double>(preds[idx]) - static_cast<double>(t_val);
        } else {
            // Read target from constant memory for double
            double t_val = const_tgts_d[idx];
            diff = preds[idx] - t_val;
        }
        thread_sum += diff * diff;
        idx += blockDim.x * gridDim.x;
    }

    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

// Host function for forward pass
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Create accumulator tensor with double precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_const", [&] {
        // Determine maximum allowed constant memory elements based on type
        int max_const = std::is_same<scalar_t, float>::value ? MAX_CONST_FLOAT : MAX_CONST_DOUBLE;

        if (num_elements > max_const) {
            // If data does not fit in constant memory, fall back to global memory kernel
            mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        } else {
            // Copy targets into constant memory
            if constexpr (std::is_same<scalar_t, float>::value) {
                cudaMemcpyToSymbol(const_tgts_f, targets.data_ptr<scalar_t>(), num_elements * sizeof(scalar_t));
            } else {
                cudaMemcpyToSymbol(const_tgts_d, targets.data_ptr<scalar_t>(), num_elements * sizeof(scalar_t));
            }
            // Launch kernel that uses constant memory for targets
            mse_forward_const_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
                predictions.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements
            );
        }
    });

    // Finalize: compute mean by dividing by the number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward with constant memory optimization (CUDA)");
}
