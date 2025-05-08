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

#define TILE_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv_transpose2d_coalesced_kernel(
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
    __shared__ float s_weight[TILE_SIZE * TILE_SIZE + TILE_SIZE]; // padded to avoid bank conflicts
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Calculate output tile position
    const int tile_row = by * TILE_SIZE;
    const int tile_col = bx * TILE_SIZE;
    
    // Load weight data into shared memory with coalesced access
    if (tx < kernel_size * kernel_size) {
        #pragma unroll
        for (int c = 0; c < in_channels; c += 4) {
            float4 w = *reinterpret_cast<const float4*>(&weight[c * kernel_size * kernel_size + tx]);
            *reinterpret_cast<float4*>(&s_weight[c * kernel_size * kernel_size + tx]) = w;
        }
    }
    __syncthreads();
    
    // Process output points with coalesced memory access
    const int out_idx = tile_row + tx;
    const int out_idy = tile_col + ty;
    
    if (out_idx < height && out_idy < width) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            const int in_row = (out_idx + padding) / stride - k;
            #pragma unroll
            for (int l = 0; l < kernel_size; l++) {
                const int in_col = (out_idy + padding) / stride - l;
                
                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                    // Load input data in vectors of 4 for better memory throughput
                    float4 in_val = *reinterpret_cast<const float4*>(
                        &input[in_row * width * in_channels + in_col * in_channels]);
                    
                    // Compute partial sums using vectorized operations
                    float4 weight_val = *reinterpret_cast<const float4*>(
                        &s_weight[k * kernel_size * in_channels + l * in_channels]);
                    
                    sum.x += in_val.x * weight_val.x;
                    sum.y += in_val.y * weight_val.y;
                    sum.z += in_val.z * weight_val.z;
                    sum.w += in_val.w * weight_val.w;
                }
            }
        }
        
        // Store results with coalesced writes
        const int out_offset = (out_idx * width + out_idy) * out_channels;
        *reinterpret_cast<float4*>(&output[out_offset]) = sum;
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
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((width + TILE_SIZE - 1) / TILE_SIZE, 
                (height + TILE_SIZE - 1) / TILE_SIZE);
    
    auto output = torch::zeros_like(x);
    
    conv_transpose2d_coalesced_kernel<<<blocks, threads>>>(
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