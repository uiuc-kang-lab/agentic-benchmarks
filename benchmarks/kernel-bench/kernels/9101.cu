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
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;
    int n = bz / out_channels;
    int oc = bz % out_channels;

    if (h_out < out_height && w_out < out_width) {
        float sum = 0.0f;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int h_in = (h_out + pad_h - kh) / stride_h;
                    int w_in = (w_out + pad_w - kw) / stride_w;

                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;

                        if (tx < TILE_SIZE && ty < TILE_SIZE) {
                            shared_input[ty * TILE_SIZE + tx] = input[input_idx];
                            shared_weight[ty * TILE_SIZE + tx] = weight[weight_idx];
                        }
                        __syncthreads();

                        sum += shared_input[ty * TILE_SIZE + tx] * shared_weight[ty * TILE_SIZE + tx];
                        __syncthreads();
                    }
                }
            }
        }

        int out_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
        output[out_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int in_height = input_size[2];
    int in_width = input_size[3];
    int out_channels = weight_size[1];
    int kernel_height = weight_size[2];
    int kernel_width = weight_size[3];
    
    int out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height;
    int out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
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
        stride[0],
        stride[1],
        padding[0],
        padding[1],
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