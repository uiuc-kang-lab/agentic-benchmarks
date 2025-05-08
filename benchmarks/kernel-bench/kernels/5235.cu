#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int constant_kernel_params[5]; // Array for kernel_size, stride, padding, dilation, input_length

__global__ void max_pool1d_kernel_constant(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int output_length,
    const bool return_indices) {

    // Load parameters from constant memory
    const int kernel_size = constant_kernel_params[0];
    const int stride = constant_kernel_params[1];
    const int padding = constant_kernel_params[2];
    const int dilation = constant_kernel_params[3];
    const int input_length = constant_kernel_params[4];

    const int b = blockIdx.z;
    const int c = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    const int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    for (int k = 0; k < kernel_size; ++k) {
        const int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            const float val = input[b * num_channels * input_length + c * input_length + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    const int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) indices[out_idx] = max_idx;
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
        indices = torch::empty({batch_size, num_channels, output_length}, 
                               torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    // Copy parameters to constant memory
    int host_params[5] = {static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding), static_cast<int>(dilation), static_cast<int>(input_length)};
    cudaMemcpyToSymbol(constant_kernel_params, host_params, sizeof(host_params));

    const dim3 blocks((output_length + 255) / 256, num_channels, batch_size);
    const dim3 threads(256);

    max_pool1d_kernel_constant<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward using constant memory (CUDA)");
}
