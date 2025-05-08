#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to calculate input indices
__device__ __forceinline__ bool calculate_input_indices(
    int out_x, int out_y, int kw, int kh,
    int pad_w, int pad_h, int stride_w, int stride_h,
    int input_width, int input_height,
    int& in_x, int& in_y) {
    
    in_x = out_x + pad_w - kw;
    in_y = out_y + pad_h - kh;
    
    if (in_x % stride_w != 0 || in_y % stride_h != 0)
        return false;
        
    in_x /= stride_w;
    in_y /= stride_h;
    
    return (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
}

// Device function to calculate memory offsets
__device__ __forceinline__ void calculate_offsets(
    int batch, int channel, int height, int width,
    int channels, int height_dim, int width_dim,
    int& offset) {
    
    offset = batch * channels * height_dim * width_dim +
             channel * height_dim * width_dim +
             height * width_dim +
             width;
}

// Device function for the core convolution computation
__device__ __forceinline__ float compute_conv_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int in_ch, int out_ch,
    int in_x, int in_y, int kh, int kw,
    int input_width, int input_height,
    int kernel_width,
    int in_channels, int batch_offset) {
    
    int input_offset;
    calculate_offsets(0, in_ch, in_y, in_x,
                     in_channels, input_height, input_width,
                     input_offset);
    
    int weight_offset = in_ch * out_channels * kernel_height * kernel_width +
                       out_ch * kernel_height * kernel_width +
                       kh * kernel_width + kw;
    
    return input[batch_offset + input_offset] * weight[weight_offset];
}

__global__ void conv_transpose2d_kernel_modular(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = batch_size * out_channels * output_height * output_width;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        const int w = idx % output_width;
        const int h = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % out_channels;
        const int b = idx / (output_width * output_height * out_channels);
        
        float sum = 0.0f;
        const int batch_offset = b * in_channels * input_height * input_width;
        
        #pragma unroll 4
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            #pragma unroll 4
            for (int kh = 0; kh < kernel_height; kh++) {
                #pragma unroll 4
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x, in_y;
                    if (calculate_input_indices(w, h, kw, kh,
                                             pad_w, pad_h, stride_w, stride_h,
                                             input_width, input_height,
                                             in_x, in_y)) {
                        
                        sum += compute_conv_element(
                            input, weight,
                            in_ch, c,
                            in_x, in_y, kh, kw,
                            input_width, input_height,
                            kernel_width,
                            in_channels, batch_offset);
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + 
                             kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + 
                            kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                             x.options());

    const int block_size = 256;
    const int num_elements = batch_size * out_channels * output_height * output_width;
    const int num_blocks = min((num_elements + block_size - 1) / block_size, 65535);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_modular<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Modular ConvTranspose2D forward (CUDA)");
}