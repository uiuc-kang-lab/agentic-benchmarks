#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Device function to calculate input coordinates
__device__ __forceinline__ bool calculate_input_coords(
    int d_out, int h_out, int w_out,
    int kd, int kh, int kw,
    int stride, int padding, int dilation,
    int in_depth, int in_height, int in_width,
    int& d_in, int& h_in, int& w_in) {
    
    d_in = d_out * stride - padding + kd * dilation;
    h_in = h_out * stride - padding + kh * dilation;
    w_in = w_out * stride - padding + kw * dilation;
    
    return (d_in >= 0 && d_in < in_depth &&
            h_in >= 0 && h_in < in_height &&
            w_in >= 0 && w_in < in_width);
}

// Device function to calculate input index
__device__ __forceinline__ int calculate_input_index(
    int b, int c, int d, int h, int w,
    int in_channels, int in_depth, int in_height, int in_width) {
    return ((b * in_channels + c) * in_depth + d) * in_height * in_width +
           h * in_width + w;
}

// Device function to calculate weight index
__device__ __forceinline__ int calculate_weight_index(
    int c_out, int ic, int kd, int kh, int kw,
    int in_channels_per_group, int kernel_d, int kernel_h, int kernel_w) {
    return ((c_out * in_channels_per_group + ic) * kernel_d + kd) * kernel_h * kernel_w +
           kh * kernel_w + kw;
}

// Device function to perform convolution for a single output element
__device__ __forceinline__ float compute_conv_element(
    const float* input,
    const float* weight,
    int b, int c_out, int d_out, int h_out, int w_out,
    int in_channels_per_group, int group,
    int in_channels, int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation) {
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        const int in_c = group * in_channels_per_group + ic;
        
        #pragma unroll
        for (int kd = 0; kd < kernel_d; kd++) {
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_w; kw++) {
                    int d_in, h_in, w_in;
                    if (calculate_input_coords(d_out, h_out, w_out, kd, kh, kw,
                                            stride, padding, dilation,
                                            in_depth, in_height, in_width,
                                            d_in, h_in, w_in)) {
                        
                        const int in_idx = calculate_input_index(b, in_c, d_in, h_in, w_in,
                                                               in_channels, in_depth, in_height, in_width);
                        const int weight_idx = calculate_weight_index(c_out, ic, kd, kh, kw,
                                                                    in_channels_per_group, kernel_d, kernel_h, kernel_w);
                        
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    return sum;
}

__global__ void conv3d_modular_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_size = gridDim.x * blockDim.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    for (int idx = tid; idx < total_elements; idx += stride_size) {
        const int w_out = idx % out_width;
        const int h_out = (idx / out_width) % out_height;
        const int d_out = (idx / (out_width * out_height)) % out_depth;
        const int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);
        
        const int group = c_out / (out_channels / groups);
        const int in_channels_per_group = in_channels / groups;
        
        float result = compute_conv_element(
            input, weight,
            b, c_out, d_out, h_out, w_out,
            in_channels_per_group, group,
            in_channels, in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation);
        
        if (bias != nullptr) {
            result += bias[c_out];
        }
        
        output[idx] = result;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    conv3d_modular_kernel<<<num_blocks, BLOCK_SIZE>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with modular device functions (CUDA)");
}