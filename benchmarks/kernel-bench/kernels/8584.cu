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
    extern __shared__ float shared_weight[];
    
    // Calculate thread position
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Use 32-thread warps for coalesced access
    const int warp_size = 32;
    const int warp_id = tx / warp_size;
    const int lane_id = tx % warp_size;
    
    // Align shared memory to 128 bytes
    constexpr int ALIGN_BYTES = 128;
    const int aligned_width = ((width + warp_size - 1) / warp_size) * warp_size;
    
    // Load weights into shared memory with coalesced access
    const int weights_per_thread = (kernel_size * kernel_size + warp_size - 1) / warp_size;
    #pragma unroll
    for (int i = 0; i < weights_per_thread; i++) {
        const int weight_idx = lane_id + i * warp_size;
        if (weight_idx < kernel_size * kernel_size) {
            shared_weight[weight_idx] = weight[weight_idx];
        }
    }
    __syncthreads();
    
    // Process output with coalesced memory access
    const int out_h = by * blockDim.y + ty;
    const int out_w_base = bx * warp_size + lane_id;
    
    if (out_h < height) {
        #pragma unroll 4
        for (int w_offset = 0; w_offset < aligned_width; w_offset += warp_size) {
            const int out_w = out_w_base + w_offset;
            if (out_w < width) {
                float sum = 0.0f;
                
                // Compute convolution
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    const int in_h = out_h / stride - padding + kh;
                    if (in_h >= 0 && in_h < height) {
                        #pragma unroll
                        for (int kw = 0; kw < kernel_size; kw++) {
                            const int in_w = out_w / stride - padding + kw;
                            if (in_w >= 0 && in_w < width) {
                                const float w_val = shared_weight[kh * kernel_size + kw];
                                const float in_val = input[in_h * width + in_w];
                                sum += w_val * in_val;
                            }
                        }
                    }
                }
                
                // Write output with coalesced access
                output[out_h * width + out_w] = sum;
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
    const int channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);
    const int kernel_size = weight.size(2);
    
    const dim3 block_size(256, 4);
    const dim3 grid_size((width + 31) / 32, (height + 3) / 4);
    const int shared_mem_size = kernel_size * kernel_size * sizeof(float);
    
    auto output = torch::zeros_like(x);
    
    conv_transpose2d_coalesced_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        channels,
        height,
        width,
        kernel_size,
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