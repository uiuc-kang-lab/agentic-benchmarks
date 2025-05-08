#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using shared memory to load a sample's feature vector and reduce global memory accesses
// Each block processes one sample (one offset within a batch) in a grid-stride loop.
// The kernel loads the data into shared memory, computes the sum of squares with a cooperative reduction,
// computes the RMS value, and then normalizes the feature vector using the shared copy.

template <typename scalar_t>
__global__ void rms_norm_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Total number of samples to process
    int total_samples = batch_size * numel_per_batch;

    // Process samples in a grid-stride loop: each block handles one sample at a time
    for (int sample = blockIdx.x; sample < total_samples; sample += gridDim.x) {
        // Determine the batch and offset within the batch
        int batch_id = sample / numel_per_batch;
        int offset = sample % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;

        // Dynamically allocated shared memory layout:
        // First: an array to store the sample's feature vector of length num_features
        // Second: an array for partial sums (one element per thread in the block)
        extern __shared__ char shared[]; 
        scalar_t* s_feature = reinterpret_cast<scalar_t*>(shared);
        scalar_t* s_sum = s_feature + num_features;

        // Each thread loads one or more elements of the sample's feature vector from global memory
        scalar_t partial = 0;
        for (int idx = threadIdx.x; idx < num_features; idx += blockDim.x) {
            int index = batch_offset + idx * numel_per_batch + offset;
            scalar_t val = input[index];
            s_feature[idx] = val;  // store the value in shared memory for reuse
            partial += val * val;  // accumulate square
        }
        
        // Store each thread's partial sum into shared memory for reduction
        s_sum[threadIdx.x] = partial;
        __syncthreads();

        // Reduce the partial sums to compute the total sum of squares for the sample
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Thread 0 computes the RMS from the total sum
        scalar_t rms = sqrt(s_sum[0] / static_cast<scalar_t>(num_features) + static_cast<scalar_t>(eps));
        __syncthreads();

        // Each thread writes back the normalized value using the shared feature data
        for (int idx = threadIdx.x; idx < num_features; idx += blockDim.x) {
            int index = batch_offset + idx * numel_per_batch + offset;
            output[index] = s_feature[idx] / rms;
        }
        __syncthreads();
    }
}


// Host function launching the shared-memory kernel
// Each block processes one sample (one offset in the batch), and grid-stride looping is used
// Shared memory per block is allocated as: (num_features + threads) * sizeof(scalar_t)

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Total number of samples to process
    int total_samples = batch_size * numel_per_batch;

    // Choose block size; each block cooperatively processes one sample
    const int threads = 256;

    // Use a grid-stride loop: set grid to a fixed size (e.g., 1024) if there are many samples
    int blocks = (total_samples < 1024) ? total_samples : 1024;

    // Compute the amount of dynamic shared memory required per block
    size_t shared_mem_bytes = (num_features + threads) * sizeof(at::ScalarTypeTraits<float>::underlying_type);
    
    // Launch the kernel using AT_DISPATCH_FLOATING_TYPES_AND_HALF to support multiple types
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_shared", ([&] {
        rms_norm_kernel_shared<scalar_t><<<blocks, threads, (num_features + threads) * sizeof(scalar_t) >>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA) using shared memory");
}
