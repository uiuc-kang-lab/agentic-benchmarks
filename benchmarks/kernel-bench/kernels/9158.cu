#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

#define TILE_SIZE 16

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_height,
    const int out_width
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int grid_z = blockIdx.z;
    int oc = grid_z % out_channels;
    int batch_id = grid_z / out_channels;
    int out_x = block_x * TILE_SIZE + tx;
    int out_y = block_y * TILE_SIZE + ty;
    float sum = 0.0f;
    
    for(int oc = 0; oc < out_channels; oc++) {
        for(int ic = 0; ic < in_channels; ic++) {
            for(int kh = 0; kh < kernel_height; kh++) {
                for(int kw = 0; kw < kernel_width; kw++) {
                    if(out_y < out_height && out_x < out_width) {
                        const int in_y = (out_y + pad_h - kh) / stride_h;
                        const int in_x = (out_x + pad_w - kw) / stride_w;
                        
                        if(in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            const float in_val = input[((batch_id * in_channels + ic) * in_height + in_y) * in_width + in_x];
                            const float w_val = weight[((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw];
                            sum += in_val * w_val;
                        }
                    }
                }
            }
        }
    }
    
    if(out_y < out_height && out_x < out_width) {
        output[((batch_id * out_channels) * out_height + out_y) * out_width + out_x] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    
    const auto out_channels = weight.size(1);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto stride_h = stride[0];
    const auto stride_w = stride[1];
    const auto pad_h = padding[0];
    const auto pad_w = padding[1];
    
    const auto out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_height;
    const auto out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    const dim3 blocks((out_width + TILE_SIZE - 1) / TILE_SIZE * ((out_height + TILE_SIZE - 1) / TILE_SIZE), batch_size);
    const dim3 threads(BLOCK_SIZE);
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_height,
        out_width
    );
    
    if (!bias_obj.is_none()) {
        auto bias = bias_obj.cast<torch::Tensor>();
        output.add_(bias.view({1, out_channels, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}