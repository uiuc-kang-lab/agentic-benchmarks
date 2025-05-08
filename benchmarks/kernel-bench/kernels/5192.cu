#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that minimizes warp divergence by precomputing valid iteration bounds
__global__ void warp_optimized_max_pool1d_kernel(
    const float* input,
    float* output,
    int64_t* indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    bool return_indices
) {
    // Each thread processes one element in the output tensor
    const int out_idx_x = blockIdx.x * blockDim.x + threadIdx.x; // output position index
    const int out_idx_y = blockIdx.y * blockDim.y + threadIdx.y; // channel index
    const int b = blockIdx.z; // batch index

    if (b >= batch_size || out_idx_y >= num_channels || out_idx_x >= output_length) return;

    // Compute the starting index in the input for this output element
    int input_start = out_idx_x * stride - padding;

    // Precompute valid k range to avoid conditional checks in the inner loop
    int k_begin = 0;
    if (input_start < 0)
        k_begin = (-input_start + dilation - 1) / dilation;  // smallest k that makes pos >= 0

    int k_end = kernel_size;
    // Compute the upper bound for k such that input_start + k*dilation < input_length
    int possible_end = (input_length - input_start + dilation - 1) / dilation;
    if (possible_end < k_end)
        k_end = possible_end;

    float max_val = -INFINITY;
    int max_idx = -1;
    int base = b * num_channels * input_length + out_idx_y * input_length;

    // Loop only over valid positions, thus avoiding divergent branching in each iteration
    for (int k = k_begin; k < k_end; ++k) {
        int pos = input_start + k * dilation;
        float val = input[base + pos];
        if (val > max_val) {
            max_val = val;
            max_idx = pos;
        }
    }

    int out_idx = b * num_channels * output_length + out_idx_y * output_length + out_idx_x;
    output[out_idx] = max_val;
    if (return_indices) {
        indices[out_idx] = max_idx;
    }
}

// Host function launching the CUDA kernel
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

    // Compute output length
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    const dim3 blocks((output_length + 31) / 32, (num_channels + 3) / 4, batch_size);
    const dim3 threads(32, 4);

    warp_optimized_max_pool1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Warp divergence optimized MaxPool1D forward (CUDA)");
}
