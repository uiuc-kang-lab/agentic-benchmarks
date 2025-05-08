#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that performs 1D max pooling. Each thread computes one output element.
// The kernel uses #pragma unroll to help the compiler optimize the loop over the pooling window.
__global__ void max_pool1d_tunable_kernel(
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

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_channels * output_length;
    if (tid >= total) return;

    // Decode the flattened index into batch, channel, and output index
    int o = tid % output_length;
    int tmp = tid / output_length;
    int c = tmp % num_channels;
    int b = tmp / num_channels;

    int input_start = o * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    int base_idx = b * num_channels * input_length + c * input_length;

    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = input[base_idx + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    int out_idx = b * num_channels * output_length + c * output_length + o;
    output[out_idx] = max_val;
    if (return_indices) {
        indices[out_idx] = max_idx;
    }
}

// Host function that selects an optimal block size using the CUDA occupancy API
// and launches the kernel with that configuration.
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
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    int total_elements = batch_size * num_channels * output_length;

    // Use CUDA Occupancy API to select the optimal block size among candidates (e.g.,32, 64, 128, 256, 512)
    int minGridSize = 0;
    int optimalBlockSize = 0;
    cudaError_t occErr = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, 
        &optimalBlockSize, 
        max_pool1d_tunable_kernel, 
        0, 
        total_elements);
    if (occErr != cudaSuccess) {
        // Fallback in case of error
        optimalBlockSize = 256;
    }

    int numBlocks = (total_elements + optimalBlockSize - 1) / optimalBlockSize;

    max_pool1d_tunable_kernel<<<numBlocks, optimalBlockSize>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with tunable block size (CUDA)");
}
