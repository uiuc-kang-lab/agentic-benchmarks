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

__global__ void conv_transpose2d_optimized_kernel(
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
    constexpr int BLOCK_SIZE = 16; // 16x16 = 256 threads
    __shared__ float weight_shared[256]; // Shared memory for weights
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Calculate output position
    const int out_x = bx * BLOCK_SIZE + tx;
    const int out_y = by * BLOCK_SIZE + ty;
    
    // Pre-load weights into shared memory
    if (tx < kernel_size && ty < kernel_size) {
        for (int c = 0; c < min(16, in_channels); c++) {
            weight_shared[c * kernel_size * kernel_size + ty * kernel_size + tx] =
                weight[c * kernel_size * kernel_size + ty * kernel_size + tx];
        }
    }
    __syncthreads();
    
    if (out_x < (width + output_padding) && out_y < (height + output_padding)) {
        for (int n = 0; n < batch_size; n++) {
            for (int oc = 0; oc < out_channels; oc++) {
                float sum = 0.0f;
                
                #pragma unroll 3
                for (int kh = 0; kh < kernel_size; kh++) {
                    #pragma unroll 3
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int in_x = (out_x + padding - kw) / stride;
                        const int in_y = (out_y + padding - kh) / stride;
                        
                        if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                            for (int ic = 0; ic < in_channels; ic++) {
                                float in_val = input[n * in_channels * height * width +
                                                   ic * height * width +
                                                   in_y * width + in_x];
                                float w_val = ic < 16 ? 
                                    weight_shared[ic * kernel_size * kernel_size + kh * kernel_size + kw] :
                                    weight[ic * kernel_size * kernel_size + kh * kernel_size + kw];
                                sum += in_val * w_val;
                            }
                        }
                    }
                }
                
                output[n * out_channels * height * width +
                       oc * height * width +
                       out_y * width + out_x] = sum;
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
    
    const int batch_size = x.size(0);
    const int height = x.size(2);
    const int width = x.size(3);
    
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    
    auto output = torch::zeros_like(x);
    
    conv_transpose2d_optimized_kernel<<<blocks, threads>>>(
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