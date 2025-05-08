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
    }
    return result;
}

// Shared memory tile sizes
#define TILE_SIZE 16
#define KERNEL_TILE 3
#define CHANNELS_PER_BLOCK 8

__global__ void conv_transpose2d_shared_kernel(
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
    __shared__ float shared_weight[CHANNELS_PER_BLOCK][KERNEL_TILE][KERNEL_TILE];
    
    // Block handles a tile of the output and a subset of output channels
    int tile_h = blockIdx.y * TILE_SIZE;
    int tile_w = blockIdx.x * TILE_SIZE;
    int out_c_start = blockIdx.z * CHANNELS_PER_BLOCK;
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_size = blockDim.x * blockDim.y;

    // Each thread processes one output point
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    int h = tile_h + local_h;
    int w = tile_w + local_w;

    if (h >= h_out || w >= w_out) return;

    // Process each batch
    for (int n = 0; n < batch_size; n++) {
        // Process assigned output channels
        for (int oc = 0; oc < CHANNELS_PER_BLOCK && out_c_start + oc < out_channels; oc++) {
            int c = out_c_start + oc;
            int g = c / out_channels_per_group;
            int c_local = c % out_channels_per_group;
            
            float sum = 0.0f;

            // Load kernel weights into shared memory
            for (int load_idx = tid; load_idx < KERNEL_TILE * KERNEL_TILE; load_idx += block_size) {
                int kh = load_idx / KERNEL_TILE;
                int kw = load_idx % KERNEL_TILE;
                if (kernel_size == 3) {
                    shared_weight[oc][kh][kw] = __ldg(&weight[((g * in_channels_per_group) * out_channels_per_group + c_local) * 9 + kh * 3 + kw]);
                }
            }
            __syncthreads();

            // Process input tiles
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                int in_channel = g * in_channels_per_group + ic;
                
                // Load input tile into shared memory
                int h_in_base = (h + padding_h) / stride_h;
                int w_in_base = (w + padding_w) / stride_w;
                
                if (h_in_base >= 0 && h_in_base < h_in && w_in_base >= 0 && w_in_base < w_in) {
                    shared_input[local_h][local_w] = __ldg(&input[((n * in_channels + in_channel) * h_in + h_in_base) * w_in + w_in_base]);
                } else {
                    shared_input[local_h][local_w] = 0.0f;
                }
                __syncthreads();

                if (kernel_size == 3) {
                    // Manually unrolled 3x3 kernel computation
                    #pragma unroll
                    for (int kh = 0; kh < 3; kh++) {
                        int h_in_idx = h + padding_h - kh;
                        if ((h_in_idx % stride_h) == 0) {
                            h_in_idx /= stride_h;
                            if (h_in_idx >= 0 && h_in_idx < h_in) {
                                #pragma unroll
                                for (int kw = 0; kw < 3; kw++) {
                                    int w_in_idx = w + padding_w - kw;
                                    if ((w_in_idx % stride_w) == 0) {
                                        w_in_idx /= stride_w;
                                        if (w_in_idx >= 0 && w_in_idx < w_in) {
                                            sum += shared_input[h_in_idx - h_in_base + kh][w_in_idx - w_in_base + kw] * 
                                                  shared_weight[oc][kh][kw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // Add bias and write output
            if (bias != nullptr) {
                sum += __ldg(&bias[c]);
            }
            
            if (h < h_out && w < w_out) {
                output[((n * out_channels + c) * h_out + h) * w_out + w] = sum;
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
    int stride_w = stride_vec.size() > 1 ? stride_vec[1] : stride_h;
    int padding_h = padding_vec[0];
    int padding_w = padding_vec.size() > 1 ? padding_vec[1] : padding_h;
    int output_padding_h = output_padding_vec[0];
    int output_padding_w = output_padding_vec.size() > 1 ? output_padding_vec[1] : output_padding_h;

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int h_in = x.size(2);
    const int w_in = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;

    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    const int in_channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;

    // Configure kernel launch parameters
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (w_out + TILE_SIZE - 1) / TILE_SIZE,
        (h_out + TILE_SIZE - 1) / TILE_SIZE,
        (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK
    );

    conv_transpose2d_shared_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with shared memory optimizations",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}