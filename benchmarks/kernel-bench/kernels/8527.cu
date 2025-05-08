#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

// Define tile dimensions for better memory access patterns
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void conv_transpose2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int h_in,
    const int w_in,
    const int out_channels,
    const int h_out,
    const int w_out,
    const int kernel_size,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group
) {
    __shared__ float shared_input[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float shared_weight[TILE_WIDTH][TILE_WIDTH];

    // Calculate tile indices
    int tile_row = blockIdx.y * TILE_HEIGHT;
    int tile_col = blockIdx.x * TILE_WIDTH;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Calculate batch and channel indices
    int batch_idx = blockIdx.z / out_channels;
    int out_channel = blockIdx.z % out_channels;
    int group = out_channel / out_channels_per_group;
    int c_local = out_channel % out_channels_per_group;

    // Process output points within this tile
    for (int th = thread_row; th < TILE_HEIGHT && (tile_row + th) < h_out; th += blockDim.y) {
        for (int tw = thread_col; tw < TILE_WIDTH && (tile_col + tw) < w_out; tw += blockDim.x) {
            float sum = 0.0f;
            int h = tile_row + th;
            int w = tile_col + tw;

            // Compute convolution for this output point
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in_candidate = h + padding_h - kh;
                    int w_in_candidate = w + padding_w - kw;

                    if ((h_in_candidate % stride_h == 0) && (w_in_candidate % stride_w == 0)) {
                        int h_in_idx = h_in_candidate / stride_h;
                        int w_in_idx = w_in_candidate / stride_w;

                        if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                            // Process input channels for current group
                            for (int ic = 0; ic < in_channels_per_group; ic++) {
                                int in_channel = group * in_channels_per_group + ic;
                                int input_idx = ((batch_idx * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                int weight_idx = (((group * in_channels_per_group + ic) * out_channels_per_group + c_local) 
                                                * kernel_size + kh) * kernel_size + kw;
                                
                                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                            }
                        }
                    }
                }
            }

            // Add bias if present
            if (bias != nullptr) {
                sum += __ldg(&bias[out_channel]);
            }

            // Write result to output
            if ((tile_row + th) < h_out && (tile_col + tw) < w_out) {
                int output_idx = ((batch_idx * out_channels + out_channel) * h_out + h) * w_out + w;
                output[output_idx] = sum;
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

    int stride_h = stride_vec[0];
    int stride_w = (stride_vec.size() > 1) ? stride_vec[1] : stride_h;
    int padding_h = padding_vec[0];
    int padding_w = (padding_vec.size() > 1) ? padding_vec[1] : padding_h;
    int output_padding_h = output_padding_vec[0];
    int output_padding_w = (output_padding_vec.size() > 1) ? output_padding_vec[1] : output_padding_h;

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int h_in = x.size(2);
    const int w_in = x.size(3);
    
    const int kernel_size = weight.size(2);
    int out_channels = weight.size(1) * groups;
    
    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    // Configure kernel launch parameters for tiled execution
    dim3 block_dim(16, 16);  // Thread block dimensions
    dim3 grid_dim(
        (w_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (h_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output_tensor.data_ptr<float>();

    conv_transpose2d_tiled_kernel<<<grid_dim, block_dim>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        h_in,
        w_in,
        out_channels,
        h_out,
        w_out,
        kernel_size,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with tiled implementation",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}