#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

// Maximum number of features that can be stored in constant memory
#define MAX_FEATURES_CONST 1024

// Constant memory for precomputed offsets
__constant__ int d_offsets[MAX_FEATURES_CONST];

// Combined kernel: uses a stride loop for workload distribution and optionally constant memory for precomputed offsets
template <typename scalar_t>
__global__ void rms_norm_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps,
    const bool use_const  // if true, use precomputed offsets from constant memory
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Each thread processes several positions using a stride loop
    for (int index = tid; index < batch_size * numel_per_batch; index += total_threads) {
        int batch_id = index / numel_per_batch;
        int offset_in_batch = index % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = 0;

        // Accumulate sum of squares for the input values of the current element across features
        if (use_const) {
            // Use constant memory offsets to avoid multiplication in inner loop
            for (int feat = 0; feat < num_features; feat++) {
                int pos = batch_offset + d_offsets[feat] + offset_in_batch;
                scalar_t val = input[pos];
                sumsq += val * val;
            }
        } else {
            // Compute offsets on the fly if num_features exceeds our constant memory limit
            for (int feat = 0; feat < num_features; feat++) {
                int pos = batch_offset + feat * numel_per_batch + offset_in_batch;
                scalar_t val = input[pos];
                sumsq += val * val;
            }
        }

        // Compute the root-mean-square value
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize each input value and write to output
        if (use_const) {
            for (int feat = 0; feat < num_features; feat++) {
                int pos = batch_offset + d_offsets[feat] + offset_in_batch;
                output[pos] = input[pos] / rms;
            }
        } else {
            for (int feat = 0; feat < num_features; feat++) {
                int pos = batch_offset + feat * numel_per_batch + offset_in_batch;
                output[pos] = input[pos] / rms;
            }
        }
    }
}

// Forward function which prepares inputs and launches the combined kernel
// It precomputes feature offsets and copies them to constant memory if possible

torch::Tensor rms_norm_cuda_forward_combined(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    // Calculate the number of elements per batch for dimensions beyond the first two
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Decide whether to use constant memory for precomputed offsets
    bool use_const = (num_features <= MAX_FEATURES_CONST);
    if (use_const) {
        // Create host-side offsets: offset for each feature is feat * numel_per_batch
        std::vector<int> offsets(num_features);
        for (int i = 0; i < num_features; i++) {
            offsets[i] = i * numel_per_batch;
        }
        // Copy offsets to constant memory
        cudaMemcpyToSymbol(d_offsets, offsets.data(), num_features * sizeof(int));
    }

    // Define execution configuration
    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_combined", ([&] {
        rms_norm_kernel_combined<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps,
            use_const
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_combined,
          "RMS normalization forward combining stride loop with constant memory optimization (CUDA)");
}
