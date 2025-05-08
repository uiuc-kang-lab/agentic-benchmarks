#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Reduced constant memory size to fit within 64KB limit
__constant__ float const_weight[16000];  // ~64KB

// CUDA kernel for transposed 3D convolution
template <int BLOCK_SIZE>
__global__ void conv_transpose3d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int padding_d, const int padding_h, const int padding_w,
    const int output_padding_d, const int output_padding_h, const int output_padding_w,
    const int groups) {
    
    const int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    const int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const int out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    const int channels_per_group = out_channels / groups;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    while (idx < total_elements) {
        const int w = idx % out_width;
        const int h = (idx / out_width) % out_height;
        const int d = (idx / (out_width * out_height)) % out_depth;
        const int c = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);
        
        const int group_idx = c / channels_per_group;
        float sum = 0.0f;
        
        // Compute valid input region for this output position
        const int in_d_start = (d + padding_d - kernel_d + 1 + stride_d - 1) / stride_d;
        const int in_h_start = (h + padding_h - kernel_h + 1 + stride_h - 1) / stride_h;
        const int in_w_start = (w + padding_w - kernel_w + 1 + stride_w - 1) / stride_w;
        
        const int in_d_end = min((d + padding_d) / stride_d + 1, in_depth);
        const int in_h_end = min((h + padding_h) / stride_h + 1, in_height);
        const int in_w_end = min((w + padding_w) / stride_w + 1, in_width);
        
        for (int id = in_d_start; id < in_d_end; id++) {
            for (int ih = in_h_start; ih < in_h_end; ih++) {
                for (int iw = in_w_start; iw < in_w_end; iw++) {
                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        const int kd = d - id * stride_d + padding_d;
                        const int kh = h - ih * stride_h + padding_h;
                        const int kw = w - iw * stride_w + padding_w;
                        
                        if (kd >= 0 && kd < kernel_d && kh >= 0 && kh < kernel_h && kw >= 0 && kw < kernel_w) {
                            const int in_ch_start = (c % channels_per_group) * (in_channels / groups);
                            const int in_ch_end = (c % channels_per_group + 1) * (in_channels / groups);
                            
                            for (int ic = in_ch_start; ic < in_ch_end; ic++) {
                                const int weight_idx = ((c * in_channels / groups + ic % (in_channels / groups)) * 
                                                      kernel_d * kernel_h * kernel_w + 
                                                      kd * kernel_h * kernel_w + 
                                                      kh * kernel_w + kw);
                                                      
                                const int input_idx = ((b * in_channels + ic) * 
                                                     in_depth * in_height * in_width + 
                                                     id * in_height * in_width + 
                                                     ih * in_width + iw);
                                                     
                                sum += input[input_idx] * const_weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        output[idx] = sum;
        idx += blockDim.x * gridDim.x;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }

    auto weight_size = weight.numel() * sizeof(float);
    TORCH_CHECK(weight_size <= 16000 * sizeof(float), 
               "Weight tensor too large for constant memory. Size: ", weight_size, " bytes");

    // Copy weight data to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight_size);

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_depth = x.size(2);
    const int in_height = x.size(3);
    const int in_width = x.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0];
    const int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1];
    const int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2];

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width},
                             x.options());

    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_depth * out_height * out_width + threads - 1) / threads;

    conv_transpose3d_kernel<256><<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups
    );

    if (bias.has_value()) {
        output.add_(*bias.value().view({1, out_channels, 1, 1, 1}));
    }

    return output;
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward with constant memory (CUDA)");
}