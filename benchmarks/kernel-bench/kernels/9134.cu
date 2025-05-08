#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__constant__ float c_weight[16384];

// Optimized kernel with larger block size and better work distribution
template<int BLOCK_SIZE = 128>
__global__ void conv_transpose2d_forward_kernel_optimized(
    const float* __restrict__ input,
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
    const int pW,
    const int elements_per_thread
) {
    // Calculate global thread ID
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total_threads = gridDim.x * BLOCK_SIZE;
    const int total_elements = N * C_out * H_out * W_out;

    // Each thread processes multiple elements
    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c_out = (idx / (W_out * H_out)) % C_out;
        const int n = idx / (W_out * H_out * C_out);

        float sum = 0.0f;

        // Unroll loops for better instruction-level parallelism
        #pragma unroll 4
        for (int ic = 0; ic < C_in; ++ic) {
            #pragma unroll
            for (int kh = 0; kh < kH; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < kW; ++kw) {
                    const int i_val = h_out + pH - kh;
                    const int j_val = w_out + pW - kw;

                    if ((i_val % sH == 0) && (j_val % sW == 0)) {
                        const int i_in = i_val / sH;
                        const int j_in = j_val / sW;

                        if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                            const int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                            const int weight_idx = ((ic * C_out + c_out) * kH + kh) * kW + kw;
                            sum += input[input_idx] * c_weight[weight_idx];
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }

        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const int weight_size = weight.numel() * sizeof(float);
    if (weight_size > 64 * 1024) {
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (!bias_obj.is_none()) {
            bias = bias_obj.cast<torch::Tensor>();
        }
        return at::conv_transpose2d(x, weight, bias, stride, padding);
    }

    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_size);

    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int C_out = weight.size(1);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    const int sH = stride[0];
    const int sW = stride[1];
    const int pH = padding[0];
    const int pW = padding[1];

    const int H_out = (H_in - 1) * sH - 2 * pH + kH;
    const int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Calculate optimal block and grid configuration
    constexpr int BLOCK_SIZE = 128;
    const int total_elements = N * C_out * H_out * W_out;
    const int elements_per_thread = 4; // Each thread processes 4 elements
    const int num_blocks = (total_elements + (BLOCK_SIZE * elements_per_thread) - 1) / (BLOCK_SIZE * elements_per_thread);

    // Launch kernel with optimized configuration
    conv_transpose2d_forward_kernel_optimized<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW, sH, sW, pH, pW,
        elements_per_thread
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with optimized block size",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}