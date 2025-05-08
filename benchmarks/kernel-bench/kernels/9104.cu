#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__constant__ float const_weights[8192];  // Assuming weights fit within this size

// Kernel optimized with constant memory for weights
__global__ void conv_transpose2d_forward_kernel_constant(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int kH,
    int kW,
    int sH,
    int sW,
    int pH,
    int pW
) {
    int out_idx = blockIdx.x;
    int ow = out_idx % W_out;
    int oh = (out_idx / W_out) % H_out;
    int oc = (out_idx / (W_out * H_out)) % C_out;
    int n  = out_idx / (W_out * H_out * C_out);

    int L = C_in * kH * kW;
    float local_sum = 0.0f;

    for (int r = threadIdx.x; r < L; r += blockDim.x) {
        int ic = r / (kH * kW);
        int rem = r % (kH * kW);
        int kh = rem / kW;
        int kw = rem % kW;

        int i_val = oh + pH - kh;
        int j_val = ow + pW - kw;

        if ((i_val % sH == 0) && (j_val % sW == 0)) {
            int i_in = i_val / sH;
            int j_in = j_val / sW;
            if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                int input_index = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                int weight_index = ((ic * C_out + oc) * kH + kh) * kW + kw;
                local_sum += input[input_index] * const_weights[weight_index];
            }
        }
    }

    __shared__ float shared_sum[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    local_sum += __shfl_down_sync(0xffffffff, local_sum, 1);
    if (lane == 0) shared_sum[warpId] = local_sum;
    __syncthreads();

    if (warpId == 0) {
        local_sum = (lane < blockDim.x / 32 ? shared_sum[lane] : 0.0f);
        local_sum += __shfl_down_sync(0xffffffff, local_sum, 1);
        if (lane == 0) {
            if (bias != nullptr) {
                local_sum += bias[oc];
            }
            int output_index = ((n * C_out + oc) * H_out + oh) * W_out + ow;
            output[output_index] = local_sum;
        }
    }
}

void load_weights_to_constant_memory(const torch::Tensor& weight) {
    cudaMemcpyToSymbol(const_weights, weight.data_ptr<float>(), weight.numel() * sizeof(float));
}

// Host function for Conv Transposed 2D forward pass
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    torch::Tensor bias = torch::Tensor();
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }
    const float* bias_ptr = (bias.defined() ? bias.data_ptr<float>() : nullptr);

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int kH = weight.size(2);
    int kW = weight.size(3);
    int C_out = weight.size(1);

    int sH = stride[0];
    int sW = stride[1];
    int pH = padding[0];
    int pW = padding[1];

    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    load_weights_to_constant_memory(weight);

    int total_outputs = N * C_out * H_out * W_out;
    int threads = 256;

    conv_transpose2d_forward_kernel_constant<<<total_outputs, threads>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with constant memory for weights",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}