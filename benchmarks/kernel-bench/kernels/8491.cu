#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int out_height,
    const int out_width
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid / out_channels;
    const int oc = bid % out_channels;
    
    // Pre-load bias if present
    float acc = (bias != nullptr) ? bias[oc] : 0.0f;
    
    for (int ih = tid; ih < in_height; ih += blockDim.x) {
        for (int iw = 0; iw < in_width; iw++) {
            const float in_val = input[n * in_channels * in_height * in_width + 
                                     ih * in_width + iw];
            
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int out_h = ih * stride - padding + kh + output_padding;
                    const int out_w = iw * stride - padding + kw + output_padding;
                    
                    if (out_h >= 0 && out_h < out_height && 
                        out_w >= 0 && out_w < out_width) {
                        const float weight_val = weight[oc * kernel_size * kernel_size + 
                                                      kh * kernel_size + kw];
                        const int out_idx = n * out_channels * out_height * out_width +
                                          oc * out_height * out_width +
                                          out_h * out_width + out_w;
                        atomicAdd(output + out_idx, in_val * weight_val);
                    }
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto kernel_size = weight.size(2);
    const auto out_channels = weight.size(1) * groups;
    
    const auto out_height = (in_height - 1) * stride[0] - 2 * padding[0] + 
                            kernel_size + output_padding[0];
    const auto out_width = (in_width - 1) * stride[1] - 2 * padding[1] + 
                           kernel_size + output_padding[1];
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                             x.options());
    
    const int threads = 256;
    const int blocks = batch_size * out_channels;
    const int shared_mem_size = threads * sizeof(float);
    
    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride[0],
        padding[0],
        output_padding[0],
        out_height,
        out_width
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with atomic optimization");
}