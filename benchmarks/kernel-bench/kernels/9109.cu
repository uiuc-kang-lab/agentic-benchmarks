/*
Hybrid ConvTranspose2D Kernel
This implementation combines a custom CUDA kernel with a fallback to PyTorch's highly optimized ATen conv_transpose2d
implementation. For small kernel sizes and when using float tensors (and contiguous data), the custom kernel with unrolled loops
is used. Otherwise, it falls back to ATen's conv_transpose2d (which can use cuDNN) for maximum efficiency.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

namespace py = pybind11;

// Custom CUDA kernel for ConvTranspose2D with loop unrolling
// Templated block size for flexibility

template <int BLOCK_SIZE = 256>
__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int sH,
    const int sW,
    const int pH,
    const int pW
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int output_idx = bid * BLOCK_SIZE + tid;

    if (output_idx >= N * C_out * H_out * W_out) return;

    // Calculate corresponding indices in the output tensor
    const int ow = output_idx % W_out;
    const int oh = (output_idx / W_out) % H_out;
    const int oc = (output_idx / (W_out * H_out)) % C_out;
    const int n  = output_idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    
    #pragma unroll
    for (int ic = 0; ic < C_in; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                int i_val = oh + pH - kh;
                int j_val = ow + pW - kw;
                if ((i_val % sH == 0) && (j_val % sW == 0)) {
                    int i_in = i_val / sH;
                    int j_in = j_val / sW;
                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[output_idx] = sum;
}


// Hybrid conv_transpose2d forward function
// Depending on the tensor types and kernel size, chooses either the custom CUDA kernel
// or falls back to ATen's conv_transpose2d implementation (which leverages cuDNN if available).

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Decide whether to use the custom kernel
    bool use_custom_kernel = true;
    // Only handle float tensors and 4D inputs for our custom kernel
    if (x.scalar_type() != at::ScalarType::Float || weight.scalar_type() != at::ScalarType::Float) {
        use_custom_kernel = false;
    }
    if (x.dim() != 4 || weight.dim() != 4) {
        use_custom_kernel = false;
    }
    
    // Use custom kernel only for small kernel sizes (arbitrarily chosen threshold)
    int kH = weight.size(2);
    int kW = weight.size(3);
    if (kH > 5 || kW > 5) {
        use_custom_kernel = false;
    }

    if (!use_custom_kernel) {
        // Fallback to ATen implementation which uses optimized libraries like cuDNN
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (!bias_obj.is_none()) {
            bias = bias_obj.cast<torch::Tensor>();
        }
        return at::conv_transpose2d(x, weight, bias, stride, padding);
    }
    
    // Prepare bias pointer if available
    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    // Extract dimensions
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int C_out = weight.size(1);
    
    int sH = stride[0];
    int sW = stride[1];
    int pH = padding[0];
    int pW = padding[1];

    // Calculate output dimensions for conv_transpose2d
    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Compute grid and block sizes
    int total_elements = N * C_out * H_out * W_out;
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch the custom kernel on the current CUDA stream
    conv_transpose2d_forward_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        sH, sW,
        pH, pW
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Hybrid Conv Transpose 2D forward: custom kernel for small kernels, fallback to ATen",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
