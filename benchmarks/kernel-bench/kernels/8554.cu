#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void conv_transposed2d_tiled_kernel(
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
    const int kernel_size,
    const int stride,
    const int padding,
    const int groups
) {
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int bz = blockIdx.z;
    
    const int batch_idx = bz / out_channels;
    const int out_ch = bz % out_channels;
    
    float acc = 0.0f;
    
    const int group_idx = out_ch / (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;
    const int channel_start = group_idx * in_channels_per_group;
    const int channel_end = (group_idx + 1) * in_channels_per_group;

    // Loop over input channels in the same group
    for (int ic = channel_start; ic < channel_end; ic++) {
        // Load input tile to shared memory
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            const int in_x = bx + tx;
            const int in_y = by + ty;
            if (in_x < in_w && in_y < in_h) {
                shared_input[ty][tx] = input[
                    ((batch_idx * in_channels + ic) * in_h + in_y) * in_w + in_x
                ];
            } else {
                shared_input[ty][tx] = 0.0f;
            }
        }
        
        // Load weight tile to shared memory
        if (tx < kernel_size && ty < kernel_size) {
            // Adjust weight indexing for groups: weight is stored as [in_channels, out_channels/groups, kernel, kernel]
            const int out_channels_per_group = out_channels / groups;
            const int local_out_ch = out_ch % out_channels_per_group;
            shared_weight[ty][tx] = weight[
                (((ic - channel_start) * out_channels_per_group + local_out_ch) * kernel_size + ty) * kernel_size + tx
            ];
        }
        
        __syncthreads();
        
        // Compute partial sum for this input channel
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            const int out_x = bx + tx;
            const int out_y = by + ty;
            
            if (out_x < out_w && out_y < out_h) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_pos_x = out_x + padding - kw;
                        int in_pos_y = out_y + padding - kh;
                        
                        if (in_pos_x % stride == 0 && in_pos_y % stride == 0) {
                            in_pos_x /= stride;
                            in_pos_y /= stride;
                            
                            if (in_pos_x >= 0 && in_pos_x < in_w &&
                                in_pos_y >= 0 && in_pos_y < in_h) {
                                acc += shared_input[in_pos_y][in_pos_x] *
                                      shared_weight[kh][kw];
                            }
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        const int out_x = bx + tx;
        const int out_y = by + ty;
        
        if (out_x < out_w && out_y < out_h) {
            const int out_idx = (
                (batch_idx * out_channels + out_ch) * out_h + out_y
            ) * out_w + out_x;
            
            if (bias != nullptr) {
                acc += bias[out_ch];
            }
            
            output[out_idx] = acc;
        }
    }
}

inline std::vector<int64_t> parseIntArrayRef(const py::object& obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    }
    return result;
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
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(0);
    
    const int out_h = (in_h - 1) * stride_vec[0] - 2 * padding_vec[0] + kernel_size + output_padding_vec[0];
    const int out_w = (in_w - 1) * stride_vec[1] - 2 * padding_vec[1] + kernel_size + output_padding_vec[1];
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    conv_transposed2d_tiled_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size,
        stride_vec[0],
        padding_vec[0],
        groups
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