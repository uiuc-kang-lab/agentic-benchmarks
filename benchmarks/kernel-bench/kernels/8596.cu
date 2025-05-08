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

template<int KERNEL_SIZE>
__global__ void conv_transpose2d_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ weight,
    float4* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int stride,
    const int padding,
    const int output_padding
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * KERNEL_SIZE * KERNEL_SIZE;

    const int tid = threadIdx.x;
    const int lid = tid % 4;  // Lane ID within float4
    const int wid = tid / 4;  // Warp ID
    
    // Load weights into shared memory using vectorized loads
    if (tid < out_channels) {
        #pragma unroll
        for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE / 4; i++) {
            float4 w = weight[tid * KERNEL_SIZE * KERNEL_SIZE / 4 + i];
            shared_weight[tid * KERNEL_SIZE * KERNEL_SIZE + i * 4 + 0] = w.x;
            shared_weight[tid * KERNEL_SIZE * KERNEL_SIZE + i * 4 + 1] = w.y;
            shared_weight[tid * KERNEL_SIZE * KERNEL_SIZE + i * 4 + 2] = w.z;
            shared_weight[tid * KERNEL_SIZE * KERNEL_SIZE + i * 4 + 3] = w.w;
        }
    }
    __syncthreads();

    // Process output points with fully unrolled loops
    #pragma unroll
    for (int n = 0; n < batch_size; n++) {
        #pragma unroll
        for (int h = 0; h < height; h += 4) {
            #pragma unroll
            for (int w = 0; w < width; w += 4) {
                float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                
                // Fully unrolled kernel loops
                #pragma unroll
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        const float weight_val = shared_weight[tid * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw];
                        const int ih = h + kh;
                        const int iw = w + kw;
                        
                        if (ih < height && iw < width) {
                            float4 in_val = input[(n * height * width + ih * width + iw) / 4];
                            sum.x += weight_val * in_val.x;
                            sum.y += weight_val * in_val.y;
                            sum.z += weight_val * in_val.z;
                            sum.w += weight_val * in_val.w;
                        }
                    }
                }
                
                // Store results using vectorized writes
                if (h < height && w < width) {
                    const int out_idx = (n * out_channels * height * width + tid * height * width + h * width + w) / 4;
                    output[out_idx] = sum;
                }
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