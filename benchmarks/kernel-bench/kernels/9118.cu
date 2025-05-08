#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Constant memory for weights
__constant__ float c_weight[16384];

template<int BLOCK_X, int BLOCK_Y>
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
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int ow = bx * BLOCK_X + tx;
    const int oh = by * BLOCK_Y + ty;
    const int oc = bz % C_out;
    const int n = bz / C_out;

    if (ow >= W_out || oh >= H_out) return;

    float sum = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < C_in; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                const int i_val = oh + pH - kh;
                const int j_val = ow + pW - kw;

                if ((i_val % sH == 0) && (j_val % sW == 0)) {
                    const int i_in = i_val / sH;
                    const int j_in = j_val / sW;

                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        const int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        const int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        sum += input[input_idx] * c_weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    const int output_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
    output[output_idx] = sum;
}

// Helper function to select optimal block dimensions
void get_optimal_block_config(
    int H_out, 
    int W_out, 
    dim3& block_dim, 
    dim3& grid_dim,
    int N,
    int C_out
) {
    // Try different block configurations based on output size
    if (H_out <= 8 && W_out <= 8) {
        block_dim = dim3(8, 8);
    } else if (H_out <= 16 && W_out <= 16) {
        block_dim = dim3(16, 8);
    } else if (H_out <= 32 && W_out <= 32) {
        block_dim = dim3(16, 16);
    } else {
        block_dim = dim3(32, 16);
    }

    // Calculate grid dimensions
    grid_dim = dim3(
        (W_out + block_dim.x - 1) / block_dim.x,
        (H_out + block_dim.y - 1) / block_dim.y,
        N * C_out
    );
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

    dim3 block_dim, grid_dim;
    get_optimal_block_config(H_out, W_out, block_dim, grid_dim, N, C_out);

    // Launch kernel with dynamic block size selection
    if (block_dim.x == 8 && block_dim.y == 8) {
        conv_transpose2d_forward_kernel<8, 8><<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            kH, kW, sH, sW, pH, pW
        );
    } else if (block_dim.x == 16 && block_dim.y == 8) {
        conv_transpose2d_forward_kernel<16, 8><<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            kH, kW, sH, sW, pH, pW
        );
    } else if (block_dim.x == 16 && block_dim.y == 16) {
        conv_transpose2d_forward_kernel<16, 16><<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            kH, kW, sH, sW, pH, pW
        );
    } else {
        conv_transpose2d_forward_kernel<32, 16><<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            kH, kW, sH, sW, pH, pW
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with optimized block sizes",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}