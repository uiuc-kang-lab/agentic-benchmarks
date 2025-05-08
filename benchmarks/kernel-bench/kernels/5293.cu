#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the max pooling operation for a given output index
__device__ inline void compute_maxpool1d(
    const float* __restrict__ input_bc,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int out_index,
    float &max_val,
    int &max_idx) {

    const int input_start = out_index * stride - padding;
    max_val = -INFINITY;
    max_idx = -1;
    
    // Loop over the kernel window
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        // Check if the position is within the valid range
        if (pos >= 0 && pos < input_length) {
            float val = __ldg(input_bc + pos);
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }
}

__global__ void max_pool1d_modular_kernel(
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

    int total_elements = batch_size * num_channels * output_length;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Decode flattened index into batch, channel, and output index
    int bc = tid / output_length;
    int out_index = tid % output_length;
    int b = bc / num_channels;
    int c = bc % num_channels;

    // Pointer to the start of the input for the current batch and channel
    const float* input_bc = input + (b * num_channels * input_length) + (c * input_length);

    float max_val;
    int max_idx;
    // Use the modular device function to compute the max value and index
    compute_maxpool1d(input_bc, input_length, kernel_size, stride, padding, dilation, out_index, max_val, max_idx);

    // Write the result to the output tensor
    int out_flat_idx = b * num_channels * output_length + c * output_length + out_index;
    output[out_flat_idx] = max_val;
    if (return_indices) {
        indices[out_flat_idx] = max_idx;
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

    int total_elements = batch_size * num_channels * output_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    max_pool1d_modular_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with modular device functions (CUDA)");
}
