#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Use constant memory for kernel weights (up to 64KB)
__constant__ float c_weight[16384];

// Kernel that processes one output element per thread. The grid and block are organized so that
// threads in a warp access consecutive elements in the output width dimension, ensuring coalesced
// global memory reads and writes. 
__global__ void conv_transpose2d_forward_kernel_coalesced(
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
    // Compute output indices: threads are arranged in 2D blocks covering H_out x W_out
    int ow = blockIdx.x * blockDim.x + threadIdx.x;  // output width index
    int oh = blockIdx.y * blockDim.y + threadIdx.y;    // output height index

    // blockIdx.z maps to the combination of batch and output channel
    int linear = blockIdx.z;
    int n  = linear / C_out;
    int oc = linear % C_out;

    if (ow < W_out && oh < H_out) {
        float sum = 0.0f;
        // Loop across input channels and kernel spatial dimensions
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int i_val = oh + pH - kh;
                    int j_val = ow + pW - kw;

                    // Check if the current location maps to a valid input (considering stride)
                    if ((i_val % sH == 0) && (j_val % sW == 0)) {
                        int i_in = i_val / sH;
                        int j_in = j_val / sW;
                        if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                            int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                            int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                            sum += input[input_idx] * c_weight[weight_idx];
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
        output[out_idx] = sum;
    }
}


// Host function to launch the kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Check if weights fit in constant memory
    int weight_size = weight.numel() * sizeof(float);
    const int max_const_size = 64 * 1024; // 64KB
    if (weight_size > max_const_size) {
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (!bias_obj.is_none()) {
            bias = bias_obj.cast<torch::Tensor>();
        }
        return at::conv_transpose2d(x, weight, bias, stride, padding);
    }

    // Copy weight to constant memory
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

    // Use 16x16 threads per block to ensure threads in a warp access consecutive memory locations in the output width
    dim3 block(16, 16);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);

    conv_transpose2d_forward_kernel_coalesced<<<grid, block>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with coalesced memory accesses",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
