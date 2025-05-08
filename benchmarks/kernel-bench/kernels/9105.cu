#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Kernel using a grid-stride loop to cover all output elements.
__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
    int total_outputs = N * C_out * H_out * W_out;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread processes multiple output elements
    for (int out_idx = tid; out_idx < total_outputs; out_idx += grid_stride) {
        // Decode flat index into (n, oc, oh, ow)
        int ow = out_idx % W_out;
        int tmp = out_idx / W_out;
        int oh = tmp % H_out;
        int tmp2 = tmp / H_out;
        int oc = tmp2 % C_out;
        int n = tmp2 / C_out;

        float acc = 0.0f;
        int reduction_length = C_in * kH * kW;

        // Iterate over all input channels and kernel positions
        for (int r = 0; r < reduction_length; r++) {
            int ic = r / (kH * kW);
            int rem = r % (kH * kW);
            int kh = rem / kW;
            int kw = rem % kW;

            int i_val = oh + pH - kh;
            int j_val = ow + pW - kw;

            // Check if the computed input coordinate aligns with the stride
            if ((i_val % sH == 0) && (j_val % sW == 0)) {
                int i_in = i_val / sH;
                int j_in = j_val / sW;
                if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                    int input_index = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                    int weight_index = ((ic * C_out + oc) * kH + kh) * kW + kw;
                    acc += input[input_index] * weight[weight_index];
                }
            }
        }

        // Apply bias if provided
        if (bias != nullptr) {
            acc += bias[oc];
        }

        int output_index = ((n * C_out + oc) * H_out + oh) * W_out + ow;
        output[output_index] = acc;
    }
}

// Host function for Conv Transposed 2D forward pass
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Determine if bias is provided
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

    // Calculate output dimensions for transposed convolution
    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    int total_outputs = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    conv_transpose2d_forward_kernel<<<blocks, threads>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with grid-stride loop optimizations",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
