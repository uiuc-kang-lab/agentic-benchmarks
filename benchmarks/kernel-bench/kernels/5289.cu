#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel precomputes the valid range of pooling indices to avoid
// divergent conditional checks inside the loop. It then uses branchless
// ternary operators to update the maximum value and its index.

__global__ void max_pool1d_uniform_kernel(
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
    const bool return_indices
) {
    // Flattened index for output element
    int total = batch_size * num_channels * output_length;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int bc = tid / output_length;
    int i  = tid % output_length;
    int b = bc / num_channels;
    int c = bc % num_channels;

    int input_start = i * stride - padding;

    // Precompute valid k range to avoid in-loop conditional branching
    int k_start = (input_start < 0) ? ((-input_start + dilation - 1) / dilation) : 0;
    int k_end_calc = (input_length - input_start + dilation - 1) / dilation;
    int k_end = (k_end_calc < kernel_size) ? k_end_calc : kernel_size;
    if (k_end < k_start) {
        k_end = k_start;  // No valid indices if window is completely out of bounds
    }

    float max_val = -INFINITY;
    int max_idx = -1;
    for (int k = k_start; k < k_end; ++k) {
        int pos = input_start + k * dilation;
        float val = input[b * num_channels * input_length + c * input_length + pos];
        // Use fmaxf for branchless max and update index using a bitwise select
        bool cond = (val > max_val);
        max_val = fmaxf(val, max_val);
        max_idx = ((-cond) & pos) | ((~-cond) & max_idx);
    }

    int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) {
        indices[out_idx] = max_idx;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices
) {
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

    int total = batch_size * num_channels * output_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    max_pool1d_uniform_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with uniform control flow (CUDA)");
}
