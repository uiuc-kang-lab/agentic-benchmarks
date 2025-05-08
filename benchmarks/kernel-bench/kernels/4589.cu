#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using stride loops to cover total workloads larger than available threads

template <typename scalar_t>
__global__ void stride_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Compute global thread index and stride (total threads in the grid)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Compute total number of elements (each element corresponds to one 'offset' per batch)
    int total_work = batch_size * numel_per_batch;

    // Stride loop: each thread processes multiple work items if necessary
    for (int t = tid; t < total_work; t += stride) {
        int batch_id = t / numel_per_batch;
        int offset_in_batch = t % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;

        // Compute sum of squares for the current position across all features
        scalar_t sumsq = static_cast<scalar_t>(0);
        for (int feat = 0; feat < num_features; ++feat) {
            int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        // Compute RMS value with epsilon for numerical stability
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize the input and write result to the output tensor
        for (int feat = 0; feat < num_features; ++feat) {
            int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Compute number of elements per batch beyond the first two dimensions
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    int total_work = batch_size * numel_per_batch;
    int threads_per_block = 256;
    int blocks = (total_work + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "stride_rms_norm_cuda", ([&] {
        stride_rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "Stride RMS normalization forward (CUDA)");
}
