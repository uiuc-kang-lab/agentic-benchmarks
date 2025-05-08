#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel minimizes warp divergence by precomputing the valid range of kernel iterations
// (k_min and k_max) so that the inner loop has a fixed iteration count without per-iteration conditionals.
// This leads to a more uniform control flow within warps.

__global__ void max_pool1d_warp_divergence_kernel(
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

    const int total_elements = batch_size * num_channels * output_length;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Compute b, c, and output index i from flattened thread id
    const int bc = tid / output_length;  // combined index for batch and channel
    const int i = tid % output_length;     // output position in the time dimension
    const int b = bc / num_channels;
    const int c = bc % num_channels;

    // Calculate the starting index in the input for the pooling window
    const int pos0 = i * stride - padding;

    // Precompute valid kernel iteration bounds to avoid divergent conditional checks inside the loop
    int k_min = 0;
    if (pos0 < 0) {
        // k_min = ceil((-pos0) / dilation)
        k_min = (-pos0 + dilation - 1) / dilation;
    }

    int k_max_valid = 0;
    if (pos0 < input_length) {
        // Maximum number of valid iterations: k such that pos0 + k*dilation <= input_length - 1
        k_max_valid = ((input_length - 1 - pos0) / dilation) + 1;
    }
    // k_end is the number of iterations we will perform; clamp it by kernel_size
    int k_end = kernel_size < k_max_valid ? kernel_size : k_max_valid;

    float max_val = -INFINITY;
    int max_idx = -1;

    // Base pointer offset for the current batch and channel
    const float* input_ptr = input + (b * num_channels * input_length + c * input_length);

    // Loop over the valid kernel window without needing per-iteration bounds checks
    for (int k = k_min; k < k_end; ++k) {
        int pos = pos0 + k * dilation;
        // Since we precomputed the valid range, pos is guaranteed to be within [0, input_length)
        float val = __ldg(input_ptr + pos);
        // Use a conditional operator to update max_val and max_idx; compilers often generate branchless code
        bool update = (val > max_val);
        max_val = update ? val : max_val;
        max_idx = update ? pos : max_idx;
    }

    // Compute the output index in the flattened output tensor (layout: [b, c, i])
    int out_idx = b * (num_channels * output_length) + c * output_length + i;
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

    const int total_elements = batch_size * num_channels * output_length;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    max_pool1d_warp_divergence_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with minimized warp divergence (CUDA)");
}
