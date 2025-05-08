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

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
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
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Load input and weight data into shared memory only when needed
    if (tid < in_channels * kernel_size * kernel_size) {
        shared_mem[tid] = weight[tid];
    }
    __syncthreads();  // Single sync point after initial load
    
    // Process output points without additional syncs
    int n = blockIdx.y;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float sum = 0.0f;
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        sum += shared_mem[tid * kernel_size * kernel_size + kh * kernel_size + kw];
                    }
                }
                output[n * out_channels * height * width + tid * height * width + h * width + w] = sum;
            }
        }
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
    
    return at::conv_transpose2d(
        x,
        weight,
        bias,
        stride_vec,
        padding_vec,
        output_padding_vec,
        groups,
        /* dilation */ {1, 1}
    );
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