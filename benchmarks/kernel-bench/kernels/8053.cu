#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {

    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Calculate global position
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Each block handles one output position for all channels in one batch
    const int batch_id = bid / output_length;
    const int out_pos = bid % output_length;
    
    // Each thread handles one output channel
    const int out_channel = tid;
    
    if (batch_id >= batch_size || out_channel >= out_channels) return;
    
    // Calculate output index
    const int out_idx = (batch_id * out_channels * output_length) + 
                       (out_channel * output_length) + out_pos;
    
    float result = 0.0f;
    
    // Compute transposed convolution
    for (int in_channel = 0; in_channel < in_channels; in_channel++) {
        for (int k = 0; k < kernel_size; k++) {
            const int in_pos = (out_pos + padding - k) / stride;
            if (in_pos >= 0 && in_pos < input_length && (out_pos + padding - k) % stride == 0) {
                const int in_idx = (batch_id * in_channels * input_length) +
                                 (in_channel * input_length) + in_pos;
                const int w_idx = (out_channel * in_channels * kernel_size) +
                                (in_channel * kernel_size) + k;
                result += input[in_idx] * weight[w_idx];
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        result += bias[out_channel];
    }
    
    output[out_idx] = result;
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int input_length = input_size[2];
    int out_channels = weight_size[0];
    int kernel_size = weight_size[2];
    
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_length},
                              x.options());

    const int shared_mem_size = (BLOCK_SIZE + kernel_size) * sizeof(float);
    const dim3 blocks((output_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 threads(BLOCK_SIZE);

    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose1d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized transposed 1D convolution forward (CUDA)");
}