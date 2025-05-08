#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Define tile dimensions for spatial workload distribution
#define TILE_W 16
#define TILE_H 16

// Utility function to parse an int or sequence of ints from a Python object
inline std::vector<int64_t> parseIntArrayRef(const py::object &obj) {
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

// Optimized CUDA kernel for ConvTranspose2d with even workload distribution
// We use a 3D grid: gridDim.z maps (batch, out_channel) and a 2D block for spatial tiling
__global__ void conv_transpose2d_even_workload_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
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
    // gridDim.z covers combined (batch, out_channel)
    int n_c = blockIdx.z;
    int n = n_c / out_channels;
    int c = n_c % out_channels;

    // Each block covers a TILE_W x TILE_H output tile
    int out_w = blockIdx.x * TILE_W + threadIdx.x;
    int out_h = blockIdx.y * TILE_H + threadIdx.y;
    if (out_w >= w_out || out_h >= h_out)
        return;

    int g = c / out_channels_per_group;
    int c_local = c % out_channels_per_group;
    float sum = 0.0f;

    // If kernel size is 3, unroll loops for faster execution
    if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int h_in_candidate = out_h + padding_h - kh;
            if (h_in_candidate % stride_h != 0) continue;
            int h_in_idx = h_in_candidate / stride_h;
            if (h_in_idx < 0 || h_in_idx >= h_in) continue;
            // Unroll kw = 0, 1, 2
            {   // kw = 0
                int kw = 0;
                int w_in_candidate = out_w + padding_w - kw;
                if (w_in_candidate % stride_w == 0) {
                    int w_in_idx = w_in_candidate / stride_w;
                    if (w_in_idx >= 0 && w_in_idx < w_in) {
                        for (int r = 0; r < in_channels_per_group; ++r) {
                            int in_channel = g * in_channels_per_group + r;
                            int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                            int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
            {   // kw = 1
                int kw = 1;
                int w_in_candidate = out_w + padding_w - kw;
                if (w_in_candidate % stride_w == 0) {
                    int w_in_idx = w_in_candidate / stride_w;
                    if (w_in_idx >= 0 && w_in_idx < w_in) {
                        for (int r = 0; r < in_channels_per_group; ++r) {
                            int in_channel = g * in_channels_per_group + r;
                            int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                            int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
            {   // kw = 2
                int kw = 2;
                int w_in_candidate = out_w + padding_w - kw;
                if (w_in_candidate % stride_w == 0) {
                    int w_in_idx = w_in_candidate / stride_w;
                    if (w_in_idx >= 0 && w_in_idx < w_in) {
                        for (int r = 0; r < in_channels_per_group; ++r) {
                            int in_channel = g * in_channels_per_group + r;
                            int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                            int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
        }
    } else {
        // Generic loop for arbitrary kernel sizes
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in_candidate = out_h + padding_h - kh;
            if (h_in_candidate % stride_h != 0) continue;
            int h_in_idx = h_in_candidate / stride_h;
            if (h_in_idx < 0 || h_in_idx >= h_in) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in_candidate = out_w + padding_w - kw;
                if (w_in_candidate % stride_w != 0) continue;
                int w_in_idx = w_in_candidate / stride_w;
                if (w_in_idx < 0 || w_in_idx >= w_in) continue;
                for (int r = 0; r < in_channels_per_group; ++r) {
                    int in_channel = g * in_channels_per_group + r;
                    int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                    int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __ldg(&bias[c]);
    }

    int output_idx = ((n * out_channels + c) * h_out + out_h) * w_out + out_w;
    output[output_idx] = sum;
}

// Forward function to set up dimensions and launch the kernel
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
    const int kernel_size = weight.size(2);  // assuming square kernel
    int out_channels = weight.size(1) * groups;

    // Calculate output dimensions
    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    // Set up 3D grid: x and y for spatial dimensions, z for combined (batch, channel)
    dim3 blockDim(TILE_W, TILE_H, 1);
    dim3 gridDim((w_out + TILE_W - 1) / TILE_W,
                 (h_out + TILE_H - 1) / TILE_H,
                 batch_size * out_channels);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output_tensor.data_ptr<float>();

    conv_transpose2d_even_workload_kernel<<<gridDim, blockDim>>>(
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

    cudaDeviceSynchronize();
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Even workload distribution for ConvTranspose2d forward kernel",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
