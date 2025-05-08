#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed read-only parameters in constant memory
__constant__ int c_num_features;
__constant__ int c_numel_per_batch;
__constant__ float c_eps;

// CUDA kernel using constant memory for parameters
template <typename scalar_t>
__global__ void rms_norm_kernel_const(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / c_numel_per_batch;
    if (batch_id >= batch_size) return;

    const int offset_in_batch = tid % c_numel_per_batch;
    const int batch_offset = batch_id * c_num_features * c_numel_per_batch;

    // Compute sum of squares for RMS normalization
    scalar_t sumsq = static_cast<scalar_t>(0);
    for (int feat = 0; feat < c_num_features; feat++) {
        const int idx = batch_offset + feat * c_numel_per_batch + offset_in_batch;
        const scalar_t val = input[idx];
        sumsq += val * val;
    }

    // Compute RMS value
    const scalar_t rms = sqrt(sumsq / c_num_features + c_eps);

    // Normalize each feature
    for (int feat = 0; feat < c_num_features; feat++) {
        const int idx = batch_offset + feat * c_numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
    }
}

// Host function: sets constant memory and launches the kernel
torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Copy frequently accessed read-only parameters to constant memory
    cudaMemcpyToSymbol(c_num_features, &num_features, sizeof(num_features));
    cudaMemcpyToSymbol(c_numel_per_batch, &numel_per_batch, sizeof(numel_per_batch));
    cudaMemcpyToSymbol(c_eps, &eps, sizeof(eps));

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_const", ([&] {
        rms_norm_kernel_const<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA) with constant memory");
}
