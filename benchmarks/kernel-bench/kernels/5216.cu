#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to perform max pooling on a slice of the input
__device__ inline void perform_max_pooling(
    const float* input,
    int base,
    int input_start,
    int kernel_size,
    int dilation,
    float &max_val,
    int &max_idx
) {
    max_val = -INFINITY;
    max_idx = -1;
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < base) {
            float val = input[base + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }
}

// Optimized max pooling kernel with modular device function
__global__ void optimized_max_pool1d_kernel(
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    int input_start = i * stride - padding;
    float max_val;
    int max_idx;
    int base = b * num_channels * input_length + c * input_length;
    
    // Use the device function to perform the max pooling operation
    perform_max_pooling(input, base, input_start, kernel_size, dilation, max_val, max_idx);

    int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) {
        indices[out_idx] = max_idx;
    }
}

// Host function wrapping the CUDA kernel launch
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

    const dim3 blocks(
        (output_length + 31) / 32,
        (num_channels + 3) / 4,
        batch_size
    );
    const dim3 threads(32, 4);

    optimized_max_pool1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Optimized MaxPool1D forward (CUDA)");
}
