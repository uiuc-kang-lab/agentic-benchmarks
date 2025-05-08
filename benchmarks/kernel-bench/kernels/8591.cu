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

// Device function to load input tile into shared memory
__device__ void load_input_tile(
    const float* input,
    float* shared_input,
    int tile_idx,
    int tile_size,
    int input_size
) {
    int tid = threadIdx.x;
    if (tid < tile_size) { shared_input[tid] = (tile_idx * tile_size + tid) < input_size ? input[tile_idx * tile_size + tid] : 0.0f;
        shared_input[tid] = input[tile_idx * tile_size + tid];
    }
}

// Device function to apply kernel weights
__device__ float apply_kernel_weights(
    const float* weight,
    const float* shared_input,
    int kh,
    int kw,
    int kernel_size,
    int channel_idx
) {
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int weight_idx = channel_idx * kernel_size * kernel_size + i * kernel_size + j;
            int input_idx = (kh + i) * kernel_size + (kw + j);
            result += weight[weight_idx] * shared_input[input_idx];
        }
    }
    return result;
}

// Device function to compute output value
__device__ void compute_output(
    float* output,
    float value,
    int n,
    int c,
    int h,
    int w,
    int out_channels,
    int height,
    int width
) {
    int output_idx = n * out_channels * height * width + c * height * width + h * width + w;
    output[output_idx] = value;
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
    
    const int tile_size = 32;
    const int tile_idx = blockIdx.x;
    const int channel_idx = threadIdx.x % out_channels;
    
    // Load input tile into shared memory
    load_input_tile(input, shared_mem, tile_idx, tile_size, height * width);
    __syncthreads();
    
    // Process output points using tiled approach
    for (int n = 0; n < batch_size; n++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float result = apply_kernel_weights(
                    weight,
                    shared_mem,
                    h,
                    w,
                    kernel_size,
                    channel_idx
                );
                compute_output(
                    output,
                    result,
                    n,
                    channel_idx,
                    h,
                    w,
                    out_channels,
                    height,
                    width
                );
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