#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define MAX_WEIGHT_SIZE 4096
__constant__ float c_weight[MAX_WEIGHT_SIZE];

__global__ void optimized_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {
    
    extern __shared__ float shared_input[];
    
    int b = blockIdx.x;
    int o = blockIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (j >= output_width) return;
    
    int group_size_out = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = o / group_size_out;
    int o_in_group = o % group_size_out;
    
    float sum = 0.0f;
    
    int tile_width = blockDim.x;
    int input_start = (j + padding - kernel_size + 1) / stride;
    int input_end = (j + padding) / stride;
    int shared_idx = threadIdx.x;
    
    if (shared_idx < (input_end - input_start + 1)) {
        int i_idx = input_start + shared_idx;
        if (i_idx >= 0 && i_idx < input_width) {
            for (int ic = 0; ic < group_in_channels; ++ic) {
                int input_channel = g * group_in_channels + ic;
                int input_index = b * (in_channels * input_width) + 
                                input_channel * input_width + i_idx;
                shared_input[ic * tile_width + shared_idx] = input[input_index];
            }
        }
    }
    __syncthreads();
    
    for (int k = 0; k < kernel_size; ++k) {
        int i_val = j + padding - k;
        if (i_val % stride != 0) continue;
        int i_idx = i_val / stride;
        if (i_idx < 0 || i_idx >= input_width) continue;
        
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int shared_index = ic * tile_width + (i_idx - input_start);
            int weight_index = (ic * group_size_out + o_in_group) * kernel_size + k;
            sum += shared_input[shared_index] * c_weight[weight_index];
        }
    }
    
    if (bias != nullptr) {
        sum += bias[o];
    }
    
    int output_index = b * (out_channels * output_width) + o * output_width + j;
    output[output_index] = sum;
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
    if (bias.has_value()) CHECK_INPUT(bias.value());
    
    int num_weight_elems = weight.numel();
    
    if (num_weight_elems > MAX_WEIGHT_SIZE) {
        return torch::conv_transpose1d(
            x, weight,
            bias.has_value() ? bias.value() : torch::Tensor(),
            stride, padding, output_padding, groups
        );
    }
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());
    
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), 
                       num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    int threads = 256;
    dim3 block(threads);
    dim3 grid(batch_size, out_channels, (output_width + threads - 1) / threads);
    
    int tile_width = threads;
    int shared_mem_size = (in_channels / groups) * tile_width * sizeof(float);
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    optimized_conv1d_kernel<<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
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
    m.def("forward", &forward, "Hybrid Transposed 1D convolution forward (CUDA)");
}