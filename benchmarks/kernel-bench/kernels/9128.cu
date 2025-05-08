#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Constant memory for weights
__constant__ float c_weight[16384];

// Kernel: each block computes one output element using warp-level reduction
// over the reduction axis (input channels * kernel height * kernel width).

__global__ void conv_transpose2d_forward_kernel_warp(
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
    // Map blockIdx.x to an output element index
    int out_idx = blockIdx.x;
    if (out_idx >= N * C_out * H_out * W_out) return;

    int ow = out_idx % W_out;
    int oh = (out_idx / W_out) % H_out;
    int oc = (out_idx / (W_out * H_out)) % C_out;
    int n  = out_idx / (W_out * H_out * C_out);

    // Total reduction work per output: over input channels and kernel positions
    int total_work = C_in * kH * kW;
    float local_sum = 0.0f;

    // Each thread in the block processes a subset of the reduction iterations
    for (int i = threadIdx.x; i < total_work; i += blockDim.x) {
        int ic = i / (kH * kW);
        int rem = i % (kH * kW);
        int kh = rem / kW;
        int kw = rem % kW;

        int i_val = oh + pH - kh;
        int j_val = ow + pW - kw;
        if ((i_val % sH == 0) && (j_val % sW == 0)) {
            int i_in = i_val / sH;
            int j_in = j_val / sW;
            if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                local_sum += input[input_idx] * c_weight[weight_idx];
            }
        }
    }

    // Intra-warp reduction using shuffle primitives
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // First thread in each warp writes the result to global memory
    if (threadIdx.x % warpSize == 0) {
        if (bias != nullptr) {
            local_sum += bias[oc];
        }
        output[out_idx] = local_sum;
    }
}

// Host function: sets up and launches the kernel

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    int weight_size = weight.numel() * sizeof(float);
    const int max_const_size = 64 * 1024;
    if (weight_size > max_const_size) {
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
    
    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    // Launch one block per output element
    const int total_output = N * C_out * H_out * W_out;
    const int BLOCK_SIZE = 256;
    dim3 grid(total_output);
    
    conv_transpose2d_forward_kernel_warp<<<grid, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW, sH, sW, pH, pW
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with warp-level reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}