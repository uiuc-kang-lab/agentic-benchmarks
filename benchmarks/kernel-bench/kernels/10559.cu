#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to compute cumulative product for a single sequence
// base_idx: Starting index for the cumulative product sequence
// dim_size: Number of elements along the cumulative product dimension
// stride: Stride to jump from one element of the sequence to the next
template <typename scalar_t>
__device__ __forceinline__ void compute_cumprod(const scalar_t* __restrict__ input,
                                                 scalar_t* __restrict__ output,
                                                 const int64_t base_idx,
                                                 const int64_t dim_size,
                                                 const int64_t stride) {
    scalar_t product = static_cast<scalar_t>(1);
    for (int i = 0; i < dim_size; i++) {
        int64_t cur_idx = base_idx + i * stride;
        product *= input[cur_idx];
        output[cur_idx] = product;
    }
}

// Kernel that assigns each thread a cumulative product sequence
// It uses grid-stride loop to cover all sequences in the tensor
template <typename scalar_t>
__global__ void cumprod_modular_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    // Each iteration processes one cumulative product series
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        // Determine batch index and starting index within the stride
        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        
        // Calculate the base index for the sequence in the flattened array
        int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        
        // Compute cumulative product for this sequence
        compute_cumprod<scalar_t>(input, output, base_idx, dim_size, stride);
    }
}

// CUDA forward function that wraps the kernel call
// 'dim' is the cumulative product dimension
torch::Tensor cumprod_cuda_modular_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Retrieve tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();

    // Determine dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    
    // Total number of independent cumulative product sequences:
    // Each sequence is a contiguous slice along 'dim'
    int64_t total_batches = input.numel() / dim_size;

    // Define kernel launch parameters
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_modular", ([&] {
        cumprod_modular_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_batches
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_modular_forward, "Cumulative product forward with modular device functions (CUDA)");
}
