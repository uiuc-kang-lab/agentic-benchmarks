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

// Constants for tile sizes
constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = 16;

__global__ void convtrans2d_tiled_kernel(
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
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];

    // 2D thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 2D block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate batch and channel indices
    const int n = bz / out_channels;
    const int c = bz % out_channels;

    // Calculate group and local channel index
    const int g = c / out_channels_per_group;
    const int c_local = c % out_channels_per_group;

    // Calculate output pixel coordinates
    const int h_out_start = by * BLOCK_SIZE;
    const int w_out_start = bx * BLOCK_SIZE;
    const int h = h_out_start + ty;
    const int w = w_out_start + tx;

    // Check if this thread should compute an output
    if (h < h_out && w < w_out) {
        float sum = 0.0f;

        // Process the convolution in tiles
        for (int tile = 0; tile < (kernel_size + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            // Load input and weight tiles into shared memory
            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh += TILE_SIZE) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw += TILE_SIZE) {
                    if ((ty + kh) < kernel_size && (tx + kw) < kernel_size) {
                        int h_in_candidate = h + padding_h - (ty + kh);
                        int w_in_candidate = w + padding_w - (tx + kw);

                        if ((h_in_candidate % stride_h == 0) && (w_in_candidate % stride_w == 0)) {
                            int h_in_idx = h_in_candidate / stride_h;
                            int w_in_idx = w_in_candidate / stride_w;

                            if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                                    int in_channel = g * in_channels_per_group + ic;
                                    int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                    int weight_idx = (((g * in_channels_per_group + ic) * out_channels_per_group + c_local) * kernel_size + (ty + kh)) * kernel_size + (tx + kw);
                                    
                                    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += __ldg(&bias[c]);
        }

        // Write output
        int output_idx = ((n * out_channels + c) * h_out + h) * w_out + w;
        output[output_idx] = sum;
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

    // Configure kernel launch parameters for 2D grid
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (w_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (h_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output_tensor.data_ptr<float>();

    convtrans2d_tiled_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "ConvTranspose2d forward with 2D tiling",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}