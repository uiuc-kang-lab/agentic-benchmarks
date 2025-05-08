#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Utility function to parse int or sequence of ints
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

// Aligned memory load helper
__device__ __forceinline__ float4 load_float4_aligned(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Shared memory declaration
extern __shared__ float shared_weights[];

__global__ void conv_transpose2d_aligned_kernel(
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
    // Load frequently used weights into shared memory
    const int tid = threadIdx.x;
    const int weights_per_thread = (kernel_size * kernel_size * in_channels_per_group + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < kernel_size * kernel_size * in_channels_per_group) {
            shared_weights[idx] = __ldg(&weight[idx]);
        }
    }
    __syncthreads();

    int total = batch_size * out_channels * h_out * w_out;
    
    // Process output elements in aligned chunks where possible
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        // Calculate output position
        const int w = index % w_out;
        int tmp = index / w_out;
        const int h = tmp % h_out;
        tmp = tmp / h_out;
        const int c = tmp % out_channels;
        const int n = tmp / out_channels;

        const int g = c / out_channels_per_group;
        const int c_local = c % out_channels_per_group;

        float sum = 0.0f;

        // Align memory accesses to 128-bit boundaries where possible
        const int aligned_w = (w & ~3);  // Align to 4-float boundary
        const bool can_use_vector = (w == aligned_w) && (w + 3 < w_out);

        if (can_use_vector && (kernel_size == 3)) {
            // Optimized path for 3x3 kernel with aligned access
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                const int h_in_candidate = h + padding_h - kh;
                if ((h_in_candidate % stride_h) == 0) {
                    const int h_in_idx = h_in_candidate / stride_h;
                    if (h_in_idx >= 0 && h_in_idx < h_in) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; kw++) {
                            const int w_in_candidate = aligned_w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                const int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx + 3 < w_in) {
                                    for (int r = 0; r < in_channels_per_group; r++) {
                                        const int in_channel = g * in_channels_per_group + r;
                                        const int input_base = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        const int weight_base = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * 3 + kh) * 3 + kw;
                                        
                                        // Vector load input data
                                        float4 input_vec = load_float4_aligned(&input[input_base]);
                                        float weight_val = shared_weights[weight_base];
                                        
                                        sum += input_vec.x * weight_val;
                                        if (w + 1 < w_out) sum += input_vec.y * weight_val;
                                        if (w + 2 < w_out) sum += input_vec.z * weight_val;
                                        if (w + 3 < w_out) sum += input_vec.w * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Generic path for non-aligned or non-3x3 cases
            for (int kh = 0; kh < kernel_size; kh++) {
                const int h_in_candidate = h + padding_h - kh;
                if ((h_in_candidate % stride_h) == 0) {
                    const int h_in_idx = h_in_candidate / stride_h;
                    if (h_in_idx >= 0 && h_in_idx < h_in) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            const int w_in_candidate = w + padding_w - kw;
                            if ((w_in_candidate % stride_w) == 0) {
                                const int w_in_idx = w_in_candidate / stride_w;
                                if (w_in_idx >= 0 && w_in_idx < w_in) {
                                    for (int r = 0; r < in_channels_per_group; r++) {
                                        const int in_channel = g * in_channels_per_group + r;
                                        const int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                                        const int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                                        sum += __ldg(&input[input_idx]) * shared_weights[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += __ldg(&bias[c]);
        }

        // Store result
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

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    const int block_size = 256;
    const int total_elements = batch_size * out_channels * h_out * w_out;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Calculate shared memory size
    const int shared_mem_size = kernel_size * kernel_size * in_channels_per_group * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output_tensor.data_ptr<float>();

    conv_transpose2d_aligned_kernel<<<grid_size, block_size, shared_mem_size>>>(
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
    m.def("forward", &forward, "ConvTranspose2d forward with aligned memory access",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}