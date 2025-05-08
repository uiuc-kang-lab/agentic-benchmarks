#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

// CUDA kernel for ConvTranspose2d that leverages shared memory for the weight (and bias) arrays
// This kernel manually unrolls the loops for kernel_size==3 and uses a generic loop for other sizes
// Shared memory is used to cache the weight tensor (and bias if provided) to reduce global memory latency

__global__ void conv_transpose2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
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
    // Allocate shared memory dynamically
    // Layout: first the entire weight tensor, then bias if not null
    extern __shared__ float shmem[];
    int weight_total = groups * in_channels_per_group * out_channels_per_group * kernel_size * kernel_size;
    int out_channels_calculated = groups * out_channels_per_group;
    float* sh_weight = shmem;
    float* sh_bias = nullptr;
    if (bias != nullptr) {
        sh_bias = shmem + weight_total;
    }

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    // Cooperative loading of weight data into shared memory
    for (int i = tid; i < weight_total; i += nthreads) {
        sh_weight[i] = weight[i];
    }
    if (sh_bias) {
        for (int i = tid; i < out_channels_calculated; i += nthreads) {
            sh_bias[i] = bias[i];
        }
    }
    __syncthreads();

    // Compute total number of output elements
    int total = batch_size * out_channels_calculated * h_out * w_out;
    for (int index = blockIdx.x * blockDim.x + tid; index < total; index += gridDim.x * nthreads) {
        // Compute output indices: w, h, c, n
        int w = index % w_out;
        int tmp = index / w_out;
        int h = tmp % h_out;
        tmp = tmp / h_out;
        int c = tmp % out_channels_calculated;
        int n = tmp / out_channels_calculated;

        int g = c / out_channels_per_group;  // group index
        int c_local = c % out_channels_per_group;
        float sum = 0.0f;

        if (kernel_size == 3) {
            // Manually unrolled loops for common 3x3 kernel
            {   // kh = 0
                int kh = 0;
                int h_in_candidate = h + padding_h - kh;
                if ((h_in_candidate % stride_h) == 0) {
                    int h_in_idx = h_in_candidate / stride_h;
                    if (h_in_idx >= 0 && h_in_idx < h_in) {
                        {   // kw = 0
                            int kw = 0;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                        {   // kw = 1
                            int kw = 1;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                        {   // kw = 2
                            int kw = 2;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            {   // kh = 1
                int kh = 1;
                int h_in_candidate = h + padding_h - kh;
                if ((h_in_candidate % stride_h) == 0) {
                    int h_in_idx = h_in_candidate / stride_h;
                    if (h_in_idx >= 0 && h_in_idx < h_in) {
                        {   // kw = 0
                            int kw = 0;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + 0;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                        {   // kw = 1
                            int kw = 1;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + 1;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                        {   // kw = 2
                            int kw = 2;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + 2;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            {   // kh = 2
                int kh = 2;
                int h_in_candidate = h + padding_h - kh;
                if ((h_in_candidate % stride_h) == 0) {
                    int h_in_idx = h_in_candidate / stride_h;
                    if (h_in_idx >= 0 && h_in_idx < h_in) {
                        {   // kw = 0
                            int kw = 0;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + 0;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                        {   // kw = 1
                            int kw = 1;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + 1;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                        {   // kw = 2
                            int kw = 2;
                            int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; ++r) {
                                        int in_channel = g * in_channels_per_group + r;
                                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + 2;
                                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Generic loop for arbitrary kernel sizes using shared memory
            for (int kh = 0; kh < kernel_size; ++kh) {
                int h_in_candidate = h + padding_h - kh;
                if ((h_in_candidate % stride_h) != 0) continue;
                int h_in_idx = h_in_candidate / stride_h;
                if (h_in_idx < 0 || h_in_idx >= h_in) continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in_candidate = w + padding_w - kw;
                    if ((w_in_candidate % stride_w) != 0) continue;
                    int w_in_idx = w_in_candidate / stride_w;
                    if (w_in_idx < 0 || w_in_idx >= w_in) continue;
                    for (int r = 0; r < in_channels_per_group; ++r) {
                        int in_channel = g * in_channels_per_group + r;
                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                        sum += __ldg(&input[input_idx]) * sh_weight[weight_idx];
                    }
                }
            }
        }
        if (sh_bias) {
            sum += sh_bias[c];
        }
        int output_idx = ((n * out_channels_calculated + c) * h_out + h) * w_out + w;
        output[output_idx] = sum;
    }
}

// Forward function: parses parameters, computes dimensions, and launches the kernel
// Allocates dynamic shared memory for caching weight (and bias if provided)

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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int h_in = x.size(2);
    int w_in = x.size(3);
    int kernel_size = weight.size(2);  // assuming square kernel
    int out_channels = weight.size(1) * groups;

    // Calculate output dimensions based on transposed convolution formula
    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    int total_elements = batch_size * out_channels * h_out * w_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    int weight_total = groups * in_channels_per_group * out_channels_per_group * kernel_size * kernel_size;
    int out_channels_calculated = groups * out_channels_per_group;
    size_t shared_mem_size = sizeof(float) * (weight_total + (bias.has_value() ? out_channels_calculated : 0));

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output_tensor.data_ptr<float>();

    conv_transpose2d_shared_kernel<<<grid_size, block_size, shared_mem_size>>>(
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
    m.def("forward", &forward, "ConvTranspose2d forward kernel using shared memory for weights and bias",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
