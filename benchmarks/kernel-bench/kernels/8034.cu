#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for weights, limited to 4KB which is 1024 floats
__constant__ float d_weights[1024];

// Kernel for 64_conv_transposed_1D with shared memory caching
__global__ void conv_transposed1d_kernel(
    const float* input,
    float* output,
    const float* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    extern __shared__ float shared_mem[];

    int batch_idx = blockIdx.x;
    int output_channel_idx = blockIdx.y;
    int output_position_idx = threadIdx.x;
    int group_idx = output_channel_idx / (out_channels / groups);

    float result = 0.0f;

    for (int kernel_pos = 0; kernel_pos < kernel_size; ++kernel_pos) {
        int input_position = output_position_idx + padding - kernel_pos;

        for (int input_channel_idx = 0; input_channel_idx < in_channels / groups; ++input_channel_idx) {
            if (input_position % stride == 0) {
                input_position /= stride;
                if (input_position >= 0 && input_position < input_width) {
                    shared_mem[threadIdx.x] = input[(batch_idx * in_channels + group_idx * (in_channels / groups) + input_channel_idx) * input_width + input_position];
                    __syncthreads();

                    result += shared_mem[threadIdx.x] * d_weights[((group_idx * (in_channels / groups) + input_channel_idx) * (out_channels / groups) + output_channel_idx) * kernel_size + kernel_pos];
                    __syncthreads();
                }
            }
        }
    }

    if (bias != nullptr) {
        result += bias[output_channel_idx];
    }

    output[(batch_idx * out_channels + output_channel_idx) * output_width + output_position_idx] = result;
}

// Host wrapper function
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

    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= 1024, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(d_weights, weight.data_ptr<float>(), num_weight_elems * sizeof(float));

    dim3 blocks(batch_size, out_channels);
    dim3 threads(output_width);

    int shared_mem_size = output_width * sizeof(float);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transposed1d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward with shared memory caching (CUDA)");
}