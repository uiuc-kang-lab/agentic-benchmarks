#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

inline std::vector<int64_t> parseIntArrayRef(const py::object &obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        int64_t val = obj.cast<int64_t>();
        result = {val, val};
    } else if (py::isinstance<py::sequence>(obj)) {
        auto seq = obj.cast<py::sequence>();
        result.reserve(2);
        for (auto item : seq) {
            result.push_back(py::cast<int64_t>(item));
        }
        if (result.size() == 1) result.push_back(result[0]);
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

template<int TILE_SIZE>
__global__ void conv_transposed2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int in_channels_per_group,
    const int out_channels_per_group
) {
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;
    
    float sum = 0.0f;
    
    const int n = bz / out_channels;
    const int oc = bz % out_channels;
    const int group = oc / out_channels_per_group;
    const int start_ic = group * in_channels_per_group;
    
    if (out_x < out_w && out_y < out_h) {
        for (int ic = start_ic; ic < start_ic + in_channels_per_group; ic += TILE_SIZE) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw += TILE_SIZE) {
                    int i_h = out_y + pad_h - kh;
                    int i_w = out_x + pad_w - kw;
                    
                    if (i_h % stride_h == 0 && i_w % stride_w == 0) {
                        i_h /= stride_h;
                        i_w /= stride_w;
                        
                        if (tx < TILE_SIZE && ty < TILE_SIZE) {
                            if (i_h >= 0 && i_h < in_h && i_w >= 0 && i_w < in_w) {
                                shared_input[ty][tx] = input[
                                    ((n * in_channels + ic) * in_h + i_h) * in_w + i_w
                                ];
                            } else {
                                shared_input[ty][tx] = 0.0f;
                            }
                            
                            if (kh * kernel_w + kw + tx < kernel_h * kernel_w) {
                                shared_weight[ty][tx] = weight[
                                    (ic * out_channels_per_group + (oc % out_channels_per_group)) 
                                    * (kernel_h * kernel_w) + kh * kernel_w + kw + tx
                                ];
                            } else {
                                shared_weight[ty][tx] = 0.0f;
                            }
                        }
                        __syncthreads();
                        
                        #pragma unroll
                        for (int k = 0; k < TILE_SIZE; ++k) {
                            sum += shared_input[ty][k] * shared_weight[tx][k];
                        }
                        __syncthreads();
                    }
                }
            }
        }
        
        if (out_x < out_w && out_y < out_h) {
            const int out_idx = (((n * out_channels + oc) * out_h) + out_y) * out_w + out_x;
            output[out_idx] = sum + (bias != nullptr ? bias[oc] : 0.0f);
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
    constexpr int TILE_SIZE = 16;
    
    if (x.numel() < 1024) {
        return at::conv_transpose2d(x, weight, bias, 
            parseIntArrayRef(stride),
            parseIntArrayRef(padding),
            parseIntArrayRef(output_padding),
            groups, {1, 1});
    }
    
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    
    int out_h = (in_h - 1) * stride_vec[0] - 2 * padding_vec[0] + kernel_h + output_padding_vec[0];
    int out_w = (in_w - 1) * stride_vec[1] - 2 * padding_vec[1] + kernel_w + output_padding_vec[1];
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv_transposed2d_kernel<TILE_SIZE><<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_vec[0],
        stride_vec[1],
        padding_vec[0],
        padding_vec[1],
        in_channels / groups,
        out_channels_per_group
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}