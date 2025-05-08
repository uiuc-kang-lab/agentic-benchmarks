#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

// Utility function to parse int or sequence of ints from a pybind11 object
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

// This kernel assumes that the input, weight, and bias tensors are 128-bit (16-byte) aligned.
// It uses __ldg() for read-only global memory accesses to leverage the read-only data cache,
// which can reduce memory latency on NVIDIA H100 GPUs.

__global__ void conv_transpose2d_kernel_ldg_aligned(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * h_out * w_out;
    if (index >= total) return;

    // Compute output spatial indices
    int w = index % w_out;
    int tmp = index / w_out;
    int h = tmp % h_out;
    tmp = tmp / h_out;
    int c = tmp % out_channels;
    int n = tmp / out_channels;

    int g = c / out_channels_per_group;
    int c_local = c % out_channels_per_group;

    float sum = 0.0f;

    // Optional: Ensure pointers are assumed to be 16-byte aligned.
    // This hint can help the compiler generate efficient aligned load instructions.
    const float *input_aligned  = input;   // assumed 128-bit aligned
    const float *weight_aligned = weight;  // assumed 128-bit aligned
    const float *bias_aligned   = bias;      // assumed 128-bit aligned

    // Loop over the kernel spatial dimensions with unrolling
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in_candidate = h + padding_h - kh;
            int w_in_candidate = w + padding_w - kw;
            if ((h_in_candidate % stride_h == 0) && (w_in_candidate % stride_w == 0)) {
                int h_in_idx = h_in_candidate / stride_h;
                int w_in_idx = w_in_candidate / stride_w;
                if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                    // Sum over the input channels in the corresponding group
                    for (int r = 0; r < in_channels_per_group; ++r) {
                        int in_channel = g * in_channels_per_group + r;
                        int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                        int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                        // Use __ldg() to load from global memory, assuming 128-bit alignment
                        float in_val = __ldg(&input_aligned[input_idx]);
                        float w_val  = __ldg(&weight_aligned[weight_idx]);
                        sum += in_val * w_val;
                    }
                }
            }
        }
    }

    if (bias_aligned != nullptr) {
        sum += __ldg(&bias_aligned[c]);
    }

    int output_idx = ((n * out_channels + c) * h_out + h) * w_out + w;
    output[output_idx] = sum;
}

// The forward function callable from PyTorch
// It sets up the convolution parameters and launches the CUDA kernel.

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

    const int kernel_size = weight.size(2); // square kernel assumed
    int out_channels = weight.size(1) * groups;

    // Compute output dimensions for transposed convolution
    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    int total_elements = batch_size * out_channels * h_out * w_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output_tensor.data_ptr<float>();

    conv_transpose2d_kernel_ldg_aligned<<<grid_size, block_size>>>(
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
    m.def("forward", &forward, "ConvTranspose2d forward with __ldg and 128-bit aligned accesses",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
