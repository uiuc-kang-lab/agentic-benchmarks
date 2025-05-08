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

// Shared memory size per block
#define TILE_SIZE 16
#define BLOCK_SIZE 256

template<typename scalar_t>
__global__ void conv_transpose2d_optimized_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
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
    
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int total_elements = batch_size * out_channels * h_out * w_out;

    // Grid-stride loop over output elements
    for (int idx = blockIdx.x * blockDim.x + tid; idx < total_elements; idx += total_threads) {
        const int w = idx % w_out;
        int tmp = idx / w_out;
        const int h = tmp % h_out;
        tmp = tmp / h_out;
        const int c = tmp % out_channels;
        const int n = tmp / out_channels;

        const int g = c / out_channels_per_group;
        const int c_local = c % out_channels_per_group;

        // Use vectorized loads where possible
        float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float sum = 0.0f;

        // Process kernel in tiles
        for (int tile_start_h = 0; tile_start_h < kernel_size; tile_start_h += TILE_SIZE) {
            for (int tile_start_w = 0; tile_start_w < kernel_size; tile_start_w += TILE_SIZE) {
                const int tile_end_h = min(tile_start_h + TILE_SIZE, kernel_size);
                const int tile_end_w = min(tile_start_w + TILE_SIZE, kernel_size);

                // Load input and weight tiles into shared memory
                if (tid < TILE_SIZE * TILE_SIZE) {
                    const int tile_h = tid / TILE_SIZE;
                    const int tile_w = tid % TILE_SIZE;
                    if (tile_h + tile_start_h < kernel_size && tile_w + tile_start_w < kernel_size) {
                        shared_input[tile_h][tile_w] = 0.0f;
                        shared_weight[tile_h][tile_w] = 0.0f;
                    }
                }
                __syncthreads();

                // Process the tile
                for (int kh = tile_start_h; kh < tile_end_h; kh++) {
                    for (int kw = tile_start_w; kw < tile_end_w; kw++) {
                        const int h_in_candidate = h + padding_h - kh;
                        const int w_in_candidate = w + padding_w - kw;

                        if ((h_in_candidate % stride_h == 0) && (w_in_candidate % stride_w == 0)) {
                            const int h_in_idx = h_in_candidate / stride_h;
                            const int w_in_idx = w_in_candidate / stride_w;

                            if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                                for (int r = 0; r < in_channels_per_group; r += 4) {
                                    if (r + 4 <= in_channels_per_group) {
                                        // Vectorized load for input and weight
                                        const int in_base = ((n * in_channels + g * in_channels_per_group + r) * h_in + h_in_idx) * w_in + w_in_idx;
                                        const int weight_base = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                                        
                                        float4 in_vec = *reinterpret_cast<const float4*>(&input[in_base]);
                                        float4 weight_vec = *reinterpret_cast<const float4*>(&weight[weight_base]);
                                        
                                        sum_vec.x += in_vec.x * weight_vec.x;
                                        sum_vec.y += in_vec.y * weight_vec.y;
                                        sum_vec.z += in_vec.z * weight_vec.z;
                                        sum_vec.w += in_vec.w * weight_vec.w;
                                    } else {
                                        // Handle remaining elements
                                        for (int ri = r; ri < in_channels_per_group; ri++) {
                                            const int in_channel = g * in_channels_per_group + ri;
                                            const int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                            const int weight_idx = (((g * in_channels_per_group + ri) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Combine vectorized and scalar sums
        sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;

        if (bias != nullptr) {
            sum += __ldg(&bias[c]);
        }

        const int output_idx = ((n * out_channels + c) * h_out + h) * w_out + w;
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
    const int out_channels = weight.size(1) * groups;
    
    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    const int total_elements = batch_size * out_channels * h_out * w_out;
    const dim3 blocks((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv_transpose2d_optimized_kernel", ([&] {
        conv_transpose2d_optimized_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
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
            in_channels / groups,
            out_channels / groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d optimized forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}