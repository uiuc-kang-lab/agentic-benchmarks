#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_depth,
    const int out_height,
    const int out_width
) {
    __shared__ float shared_input[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    int h_out = by * BLOCK_SIZE + ty;
    int w_out = bx * BLOCK_SIZE + tx;
    
    for(int n = 0; n < batch_size; n++) {
        for(int d_out = 0; d_out < out_depth; d_out++) {
            for(int oc = 0; oc < out_channels; oc++) {
                if(h_out < out_height && w_out < out_width) {
                    float sum = 0.0f;
                    
                    for(int ic = 0; ic < in_channels; ic++) {
                        for(int kd = 0; kd < kernel_d; kd++) {
                            for(int kh = 0; kh < kernel_h; kh++) {
                                for(int kw = 0; kw < kernel_w; kw++) {
                                    int h_in = (h_out + pad_h - kh) / stride_h;
                                    int w_in = (w_out + pad_w - kw) / stride_w;
                                    int d_in = (d_out + pad_d - kd) / stride_d;
                                    
                                    if(h_in >= 0 && h_in < in_height && 
                                       w_in >= 0 && w_in < in_width && 
                                       d_in >= 0 && d_in < in_depth) {
                                        int input_idx = ((n * in_channels + ic) * in_depth + d_in) * 
                                                        in_height * in_width + h_in * in_width + w_in;
                                        int weight_idx = ((oc * in_channels + ic) * kernel_d + kd) * 
                                                         kernel_h * kernel_w + kh * kernel_w + kw;
                                        
                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    int out_idx = ((n * out_channels + oc) * out_depth + d_out) * 
                                  out_height * out_width + h_out * out_width + w_out;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int in_depth = input_size[2];
    int in_height = input_size[3];
    int in_width = input_size[4];
    
    int out_channels = weight_size[1] * groups;
    int kernel_d = weight_size[2];
    int kernel_h = weight_size[3];
    int kernel_w = weight_size[4];
    
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2];
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width},
                               x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1);
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        out_depth, out_height, out_width
    );
    
    if (bias.has_value()) {
        output.add_(*bias).view({out_channels, 1, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}