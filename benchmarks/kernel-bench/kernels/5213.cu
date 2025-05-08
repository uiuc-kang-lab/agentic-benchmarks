#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel is designed to ensure memory coalescing by aligning global memory accesses.
// We use a 3D grid where blockIdx.z is the batch, blockIdx.y is the channel, and blockIdx.x*blockDim.x + threadIdx.x is the spatial output index.
// In this configuration, threads within a warp read and write consecutive memory addresses, ensuring coalesced accesses.

__global__ void coalesced_max_pool1d_kernel(
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
    bool return_indices) {

    // Determine indices based on 3D grid configuration
    int b = blockIdx.z;               // Batch index
    int c = blockIdx.y;               // Channel index
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // Output spatial index

    if (b >= batch_size || c >= num_channels || o >= output_length)
        return;

    int input_start = o * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    // Base pointer for the current (batch, channel) in the input
    int base_in = b * num_channels * input_length + c * input_length;

    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            // Use __ldg to leverage the read-only cache for efficient memory access
            float val = __ldg(&input[base_in + pos]);
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    // Compute the flattened output index
    int out_index = b * num_channels * output_length + c * output_length + o;
    output[out_index] = max_val;
    if (return_indices) {
        indices[out_index] = max_idx;
    }
}

// Host function wrapping the CUDA kernel launch

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    int batch_size = x.size(0);
    int num_channels = x.size(1);
    int input_length = x.size(2);
    
    int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");
    
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }
    
    // Configure grid and block so that threads in a warp correspond to contiguous output positions
    const int threads = 32;  // Warp size; ensures coalesced accesses along the output dimension
    dim3 block(threads);
    dim3 grid((output_length + threads - 1) / threads, num_channels, batch_size);

    coalesced_max_pool1d_kernel<<<grid, block>>>(
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
        return_indices);
        
    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced MaxPool1D forward (CUDA)");
}
