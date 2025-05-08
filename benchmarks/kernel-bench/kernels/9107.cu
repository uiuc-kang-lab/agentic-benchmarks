#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

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
    // 2D block structure for spatial locality
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Block dimensions
    const int block_width = blockDim.x;
    const int block_height = blockDim.y;
    
    // Grid dimensions
    const int grid_width = gridDim.x;
    const int grid_height = gridDim.y;
    
    // Stride calculation for output traversal
    const int h_stride = block_height * grid_height;
    const int w_stride = block_width * grid_width;
    
    // Starting position for this thread
    const int h_start = by * block_height + ty;
    const int w_start = bx * block_width + tx;
    
    // Iterate over batches and output channels
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < C_out; oc++) {
            // Strided loop over output height
            for (int oh = h_start; oh < H_out; oh += h_stride) {
                // Strided loop over output width
                for (int ow = w_start; ow < W_out; ow += w_stride) {
                    float sum = 0.0f;
                    
                    // Compute convolution for this output position
                    for (int ic = 0; ic < C_in; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                // Compute corresponding input position
                                int i_val = oh + pH - kh;
                                int j_val = ow + pW - kw;
                                
                                // Check stride alignment and bounds
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
                    
                    // Add bias if present
                    if (bias != nullptr) {
                        sum += bias[oc];
                    }
                    
                    // Write output
                    int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

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
    
    // Use 2D block configuration
    dim3 block_size(16, 16);
    dim3 grid_size(
        (W_out + block_size.x - 1) / block_size.x,
        (H_out + block_size.y - 1) / block_size.y
    );
    
    conv_transpose2d_forward_kernel<<<grid_size, block_size>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with strided loops",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}