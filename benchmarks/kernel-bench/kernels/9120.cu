#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Constant memory for weights
__constant__ float c_weight[16384];

// Balanced kernel with even workload distribution

__global__ void conv_transpose2d_forward_kernel(
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
    const int pW
) {
    extern __shared__ float shared_input[];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int ow = bx % W_out;
    const int oh = (bx / W_out) % H_out;
    const int oc = (bx / (W_out * H_out)) % C_out;
    const int n = bx / (W_out * H_out * C_out);

    float sum = 0.0f;

    for (int ic = tx; ic < C_in; ic += blockDim.x) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int i_val = oh + pH - kh;
                const int j_val = ow + pW - kw;

                if ((i_val % sH == 0) && (j_val % sW == 0)) {
                    const int i_in = i_val / sH;
                    const int j_in = j_val / sW;

                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        const int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        const int s_index = ((i_in * W_in) + j_in);
                        shared_input[s_index] = input[input_idx];

                        __syncthreads();

                        const int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        sum += shared_input[s_index] * c_weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[bx] = sum;
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
        // Fallback to cuDNN for large weights
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

    const int total_elements = N * C_out * H_out * W_out;

    // Maximum number of threads per block
    int threads_per_block = 256;
    int num_blocks = total_elements;

    size_t shared_memory_size = H_in * W_in * sizeof(float);

    conv_transpose2d_forward_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
        x.data_ptr<float>(),
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with workload balancing",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
