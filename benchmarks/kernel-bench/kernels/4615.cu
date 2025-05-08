#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Constant memory for frequently accessed, read-only parameters
__constant__ int c_num_features;
__constant__ int c_numel_per_batch;
__constant__ float c_eps;

// Kernel that uses constant memory for broadcasted parameters
template <typename scalar_t>
__global__ void rms_norm_const_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int index = tid; index < batch_size * c_numel_per_batch; index += total_threads) {
        int batch_id = index / c_numel_per_batch;
        int offset = index % c_numel_per_batch;
        int base = batch_id * c_num_features * c_numel_per_batch;

        scalar_t sumsq = static_cast<scalar_t>(0);
        #pragma unroll
        for (int feat = 0; feat < c_num_features; feat++) {
            int pos = base + feat * c_numel_per_batch + offset;
            scalar_t val = input[pos];
            sumsq += val * val;
        }

        scalar_t rms = sqrt(sumsq / c_num_features + c_eps);

        #pragma unroll
        for (int feat = 0; feat < c_num_features; feat++) {
            int pos = base + feat * c_numel_per_batch + offset;
            output[pos] = input[pos] / rms;
        }
    }
}

// CUDA forward function that sets constant memory variables and launches the kernel
torch::Tensor rms_norm_cuda_forward_const(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Copy scalar parameters to constant memory
    cudaMemcpyToSymbol(c_num_features, &num_features, sizeof(int));
    cudaMemcpyToSymbol(c_numel_per_batch, &numel_per_batch, sizeof(int));
    cudaMemcpyToSymbol(c_eps, &eps, sizeof(float));

    int total_work = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_const", ([&] {
        rms_norm_const_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_const, "RMS normalization forward using constant memory (CUDA)");
}
