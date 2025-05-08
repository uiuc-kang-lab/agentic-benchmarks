#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with shared memory tiling for coalesced global memory accesses
__global__ void max_pool1d_kernel_coalesced(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    const bool return_indices) {

    // Determine the batch and channel for this block
    const int b = blockIdx.z;
    const int c = blockIdx.y;

    // Each block handles a contiguous segment of output positions
    // blockIdx.x * blockDim.x gives the starting output index for this block
    const int out_block_start = blockIdx.x * blockDim.x;
    const int out_idx = out_block_start + threadIdx.x;

    // Compute the starting index in the input for the first output of this block
    // For an output index i, the corresponding input start is: i * stride - padding
    // For the block, the first input to load is:
    const int in_block_start = out_block_start * stride - padding;

    // Determine the size of the shared memory region needed:
    // For the block, the last thread (threadIdx.x = blockDim.x - 1) accesses up to:
    // ( (out_block_start + blockDim.x - 1) * stride - padding ) + (kernel_size - 1) * dilation
    // Hence, the shared memory size is:
    const int shared_size = (blockDim.x - 1) * stride + (kernel_size - 1) * dilation + 1;

    extern __shared__ float shm[];

    // Coalesced load of the required input region into shared memory
    for (int j = threadIdx.x; j < shared_size; j += blockDim.x) {
        int in_idx = in_block_start + j;
        if (in_idx >= 0 && in_idx < input_length) {
            shm[j] = input[b * num_channels * input_length + c * input_length + in_idx];
        } else {
            shm[j] = -INFINITY;
        }
    }
    __syncthreads();

    if (out_idx < output_length) {
        // For the current output, the local starting index in shared memory is:
        // (i * stride - padding) - (out_block_start * stride - padding) = threadIdx.x * stride
        int local_start = threadIdx.x * stride;
        float max_val = -INFINITY;
        int max_idx = -1;
        // The corresponding global input start index
        int global_in_start = out_idx * stride - padding;

        // Loop over the pooling window
        for (int k = 0; k < kernel_size; ++k) {
            int shm_index = local_start + k * dilation;
            float val = shm[shm_index];
            if (val > max_val) {
                max_val = val;
                max_idx = global_in_start + k * dilation;
            }
        }

        int out_offset = b * num_channels * output_length + c * output_length + out_idx;
        output[out_offset] = max_val;
        if (return_indices) {
            indices[out_offset] = max_idx;
        }
    }
}

// Host function to launch the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
                                 torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    // Use a 1D block to cover contiguous output positions for each (batch, channel)
    const int block_size = 256;
    const dim3 threads(block_size);
    const dim3 blocks((output_length + block_size - 1) / block_size, num_channels, batch_size);

    // Calculate dynamic shared memory size per block
    int shared_size = (block_size - 1) * stride + (kernel_size - 1) * dilation + 1;
    size_t shared_mem_bytes = shared_size * sizeof(float);

    max_pool1d_kernel_coalesced<<<blocks, threads, shared_mem_bytes>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with coalesced memory access (CUDA)");
}
