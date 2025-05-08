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
    extern __shared__ float shared_mem[];
    float* shared_output = shared_mem;
    
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    
    // Block dimensions
    const int BLOCK_SIZE = 16;
    const int h_start = by * BLOCK_SIZE;
    const int w_start = bx * BLOCK_SIZE;
    
    // Initialize shared memory
    if (tx * BLOCK_SIZE + ty < BLOCK_SIZE * BLOCK_SIZE) {
        shared_output[tx * BLOCK_SIZE + ty] = 0.0f;
    }
    __syncthreads();

    // Compute partial results in shared memory
    for (int k = 0; k < kernel_size; k++) {
        const int h_pos = h_start + tx;
        const int w_pos = w_start + ty;
        
        if (h_pos < height && w_pos < width) {
            float partial_sum = 0.0f;
            
            #pragma unroll
            for (int i = 0; i < kernel_size; i++) {
                const int weight_idx = k * kernel_size + i;
                const int input_offset = h_pos * width + w_pos;
                
                if (input_offset < height * width) {
                    partial_sum += weight[weight_idx] * input[input_offset];
                }
            }
            
            // Use shared memory atomic add to accumulate results
            atomicAdd(&shared_output[tx * BLOCK_SIZE + ty], partial_sum);
        }
    }
    __syncthreads();

    // Write results to global memory
    const int h_out = h_start + tx;
    const int w_out = w_start + ty;
    
    if (h_out < height && w_out < width) {
        const int output_idx = h_out * width + w_out;
        output[output_idx] = shared_output[tx * BLOCK_SIZE + ty];
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
    
    // Calculate grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x + 1, (height + threadsPerBlock.y - 1) / threadsPerBlock.y + 1);
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    const int shared_mem_size = 16 * 16 * sizeof(float);
    
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