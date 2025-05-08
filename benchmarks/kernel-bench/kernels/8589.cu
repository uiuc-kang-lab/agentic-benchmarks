#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

inline std::vector<int64_t> parseIntArrayRef(const py::object& obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding
) {
    __shared__ float shared_weight[TILE_SIZE * TILE_SIZE];
    __shared__ float shared_input[TILE_SIZE * TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int out_h = by * TILE_SIZE + ty;
    const int out_w = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic += TILE_SIZE) {
        if (tx < kernel_size && ty < kernel_size) {
            const int weight_idx = ic * kernel_size * kernel_size + ty * kernel_size + tx;
            if (ic + tx < in_channels && ty < kernel_size) {
                shared_weight[ty * TILE_SIZE + tx] = weight[weight_idx];
            }
        }
        
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            const int input_idx = ic * height * width + out_h * width + out_w;
            if (out_h < height && out_w < width) {
                shared_input[ty * TILE_SIZE + tx] = input[input_idx];
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            #pragma unroll
            for (int l = 0; l < kernel_size; ++l) {
                if (out_h < height && out_w < width) {
                    sum += shared_weight[k * kernel_size + l] * 
                           shared_input[(ty + k) * TILE_SIZE + (tx + l)];
                }
            }
        }
        
        __syncthreads();
    }
    
    if (out_h < height && out_w < width) {
        const int out_idx = by * TILE_SIZE * width + bx * TILE_SIZE + ty * width + tx;
        output[out_idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    
    const int batch_size = x.size(0);
    const int height = x.size(2);
    const int width = x.size(3);
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE,
                   (height + TILE_SIZE - 1) / TILE_SIZE);
    
    auto output = torch::zeros_like(x);
    
    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        x.size(1),
        weight.size(1),
        height,
        width,
        weight.size(2),
        stride_vec[0],
        padding_vec[0],
        output_padding_vec[0]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}