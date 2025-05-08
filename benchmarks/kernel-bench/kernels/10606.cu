#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define warp size for warp-level operations
#define WARP_SIZE 32

// Kernel function using shared memory and warp-level primitives
template <typename scalar_t>
__global__ void cumprod_kernel_shared_warp(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {

    extern __shared__ scalar_t shared_data[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;

    if (idx < numel / dim_size) {
        scalar_t product = 1;
        int start_idx = batch_idx * stride * dim_size + in_idx;

        // Load data into shared memory
        for (int i = 0; i < dim_size; i++) {
            int curr_idx = start_idx + i * stride;
            shared_data[threadIdx.x] = input[curr_idx];
            __syncthreads();

            // Perform cumulative product in shared memory
            for (int offset = 1; offset < blockDim.x; offset *= 2) {
                scalar_t val = (threadIdx.x >= offset) ? shared_data[threadIdx.x - offset] : 1;
                __syncthreads();
                shared_data[threadIdx.x] *= val;
                __syncthreads();
            }

            // Write result to global memory
            product *= shared_data[threadIdx.x];
            output[curr_idx] = product;
        }
    }
}

// Host function to launch the kernel
torch::Tensor cumprod_cuda_forward_shared_warp(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    int64_t total_threads = numel / dim_size;
    
    const int threads = 256;  // Use a block size that is a multiple of warp size
    const int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_shared_warp", ([&] {
        cumprod_kernel_shared_warp<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_shared_warp, "Cumulative product forward with shared memory and warp optimization (CUDA)");
}