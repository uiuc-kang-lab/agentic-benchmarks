#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Device helper: Decode flat output index into (n, oc, oh, ow)
__device__ inline void decode_output_index(int out_idx, int W_out, int H_out, int C_out, int &n, int &oc, int &oh, int &ow) {
    ow = out_idx % W_out;
    int temp = out_idx / W_out;
    oh = temp % H_out;
    temp = temp / H_out;
    oc = temp % C_out;
    n = temp / C_out;
}

// Device helper: Compute input coordinates for a given output coordinate and kernel offset
__device__ inline bool get_input_coords(int oh, int ow, int kh, int kw,
                                           int sH, int sW, int pH, int pW,
                                           int H_in, int W_in,
                                           int &i_in, int &j_in) {
    int i_val = oh + pH - kh;
    int j_val = ow + pW - kw;
    // Check if coordinates align with the stride
    if ((i_val % sH) != 0 || (j_val % sW) != 0) {
        return false;
    }
    i_in = i_val / sH;
    j_in = j_val / sW;
    return (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in);
}

// Device helper: Compute index into the input tensor
__device__ inline int get_input_index(int n, int C_in, int H_in, int W_in,
                                          int ic, int i_in, int j_in) {
    return ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
}

// Device helper: Compute index into the weight tensor
__device__ inline int get_weight_index(int C_out, int kH, int kW,
                                           int ic, int oc, int kh, int kw) {
    return ((ic * C_out + oc) * kH + kh) * kW + kw;
}

// Main CUDA kernel using grid-stride loop and modular device functions
__global__ void modular_conv_transpose2d_forward_kernel(
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
    
    for (int out_idx = tid; out_idx < total_outputs; out_idx += grid_stride) {
        int n, oc, oh, ow;
        decode_output_index(out_idx, W_out, H_out, C_out, n, oc, oh, ow);
        float sum = 0.0f;
        
        // Loop over input channels
        for (int ic = 0; ic < C_in; ++ic) {
            // Loop over kernel height and width
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int i_in, j_in;
                    if (get_input_coords(oh, ow, kh, kw, sH, sW, pH, pW, H_in, W_in, i_in, j_in)) {
                        int inp_idx = get_input_index(n, C_in, H_in, W_in, ic, i_in, j_in);
                        int w_idx = get_weight_index(C_out, kH, kW, ic, oc, kh, kw);
                        sum += input[inp_idx] * weight[w_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[out_idx] = sum;
    }
}

// Host function for ConvTranspose2D forward pass
torch::Tensor modular_conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }
    
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int C_out = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);
    int sH = stride[0];
    int sW = stride[1];
    int pH = padding[0];
    int pW = padding[1];
    
    // Compute output dimensions
    int H_out = (H_in - 1) * sH - 2 * pH + kH;
    int W_out = (W_in - 1) * sW - 2 * pW + kW;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    int total_outputs = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    modular_conv_transpose2d_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    m.def("forward", &modular_conv_transpose2d_forward, "Modular Conv Transpose 2D forward with device helper functions",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
