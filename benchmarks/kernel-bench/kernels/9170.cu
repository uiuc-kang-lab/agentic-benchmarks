#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// Tile size definitions for shared memory
#define TILE_OUT_CHANNELS 16
#define TILE_WIDTH 16

// Kernel leveraging shared memory optimized for H100 GPU
__global__ void conv_transpose2d_shared_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int in_channels, int out_channels
) {
    extern __shared__ float shared_data[];
    float* shared_input = shared_data;
    float* shared_weight = shared_data + TILE_OUT_CHANNELS * TILE_WIDTH * TILE_WIDTH;

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_channel = blockIdx.z;

    float sum = 0.0f;

    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int pad_y = -pad_h; pad_y < kernel_h - pad_h; ++pad_y) {
            for (int pad_x = -pad_w; pad_x < kernel_w - pad_w; ++pad_x) {
                int in_y = out_y * stride_h + pad_y;
                int in_x = out_x * stride_w + pad_x;

                if (in_y >= 0 && in_x >= 0 && in_y < in_height && in_x < in_width) {
                    int input_idx = in_channel * in_height * in_width + in_y * in_width + in_x;
                    shared_input[threadIdx.y * blockDim.x + threadIdx.x] = input[input_idx];
                } else {
                    shared_input[threadIdx.y * blockDim.x + threadIdx.x] = 0;
                }

                __syncthreads();

                int weight_idx = (in_channel * out_channels + out_channel) * kernel_h * kernel_w + (pad_y + pad_h) * kernel_w + (pad_x + pad_w);
                shared_weight[threadIdx.y * blockDim.x + threadIdx.x] = weight[weight_idx];

                __syncthreads();

                sum += shared_input[threadIdx.y * blockDim.x + threadIdx.x] * shared_weight[threadIdx.y * blockDim.x + threadIdx.x];

                __syncthreads();
            }
        }
    }

    if (out_x < out_width && out_y < out_height && out_channel < out_channels) {
        if (bias != nullptr) {
            sum += bias[out_channel];
        }
        int output_idx = (out_channel * out_height * out_width) + out_y * out_width + out_x;
        output[output_idx] = sum;
    }
}

// Host function
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];

    int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, input.options());

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim((out_width + block_dim.x - 1) / block_dim.x, (out_height + block_dim.y - 1) / block_dim.y, out_channels);

    size_t shared_memory_size = (TILE_OUT_CHANNELS * TILE_WIDTH * TILE_WIDTH + TILE_OUT_CHANNELS * TILE_WIDTH * TILE_WIDTH) * sizeof(float);

    for (int b = 0; b < batch; ++b) {
        const float* input_ptr = input[b].data_ptr<float>();  
        float* output_ptr = output[b].data_ptr<float>();
        const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

        conv_transpose2d_shared_optimized_kernel<<<grid_dim, block_dim, shared_memory_size>>>(
            input_ptr,
            weight.data_ptr<float>(),
            bias_ptr,
            output_ptr,
            in_height, in_width,
            out_height, out_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            in_channels, out_channels
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward optimized",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
