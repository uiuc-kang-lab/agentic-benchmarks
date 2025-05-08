#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel organizes threads so that each block processes contiguous output indices along the innermost dimension.
// Threads in the same warp will access consecutive memory locations in the output tensor, ensuring memory coalescing.

__global__ void max_pool1d_coalesced_aligned_kernel(
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

    // Use a 2D grid: blockIdx.x covers output index dimension; blockIdx.y covers (b, c) combinations.
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index in the innermost dimension
    int row = blockIdx.y;  // Flattened index for (batch, channel)
    int b = row / num_channels;
    int c = row % num_channels;

    if (i >= output_length) return;

    // Compute base pointer for the current batch and channel
    const float* input_base = input + (b * num_channels * input_length + c * input_length);
    int input_start = i * stride - padding;

    float max_val = -INFINITY;
    int max_idx = -1;

    // Loop over the pooling window
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = __ldg(input_base + pos);
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    // Compute the output index. Memory layout: [b, c, i] with i as the fastest dimension.
    int out_index = b * (num_channels * output_length) + c * output_length + i;
    output[out_index] = max_val;
    if (return_indices)
        indices[out_index] = max_idx;
}


// The forward function creates output tensors and launches the kernel with a 2D grid.
// Grid dimensions:
//   - gridDim.x: covers the output_length dimension using blockDim.x threads
//   - gridDim.y: covers the (batch, channel) combinations

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor.");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous.");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive.");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Configure a 2D grid: threads.x covers the contiguous 'i' dimension, threads in each block read/write consecutive
    // memory locations, ensuring coalescing. Block dimension in y is implicitly 1.
    const int threads_x = 256;
    dim3 threads(threads_x);
    dim3 blocks((output_length + threads_x - 1) / threads_x, (batch_size * num_channels + threads.y - 1) / threads.y);

    max_pool1d_coalesced_aligned_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with coalesced and aligned accesses (CUDA)");
}
