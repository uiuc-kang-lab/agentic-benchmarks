#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int const_params[5]; // batch_size, num_channels, input_length, kernel_size, stride

__global__ void max_pool1d_kernel_constant(
    const float* input,
    float* output,
    int64_t* indices,
    const int padding,
    const int dilation,
    const int output_length,
    bool return_indices)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (b >= const_params[0] || c >= const_params[1] || i >= output_length) return;

    const int input_start = i * const_params[4] - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    for (int k = 0; k < const_params[3]; ++k) {
        const int pos = input_start + k * dilation;
        if (pos >= 0 && pos < const_params[2]) {
            const float val = input[b * const_params[1] * const_params[2] + c * const_params[2] + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    const int out_idx = b * const_params[1] * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) indices[out_idx] = max_idx;
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices)
{
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
            options.dtype(torch::kInt64));
    }

    const int h_params[5] = {batch_size, num_channels, input_length, kernel_size, stride};
    cudaMemcpyToSymbol(const_params, h_params, sizeof(int) * 5);

    const dim3 blocks(
        (output_length + 31) / 32,
        (num_channels + 3) / 4,
        batch_size
    );
    const dim3 threads(32, 4);

    max_pool1d_kernel_constant<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        padding,
        dilation,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward (CUDA)");
}