#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__constant__ float c_weight[16384];

#define TILE_SIZE_H 8
#define TILE_SIZE_W 8
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void conv_transpose2d_forward_kernel_tiled(
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
    const int tile_start_h = blockIdx.y * TILE_SIZE_H;
    const int tile_start_w = blockIdx.x * TILE_SIZE_W;
    const int n = blockIdx.z / C_out;
    const int oc = blockIdx.z % C_out;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float partial_sums[TILE_SIZE_H/2][TILE_SIZE_W/2] = {0.0f};

    #pragma unroll
    for (int th = 0; th < TILE_SIZE_H/2; th++) {
        #pragma unroll
        for (int tw = 0; tw < TILE_SIZE_W/2; tw++) {
            const int oh = tile_start_h + th * 2 + ty % 2;
            const int ow = tile_start_w + tw * 2 + tx % 2;

            if (oh < H_out && ow < W_out) {
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

                partial_sums[th][tw] = sum;
            }
        }
    }

    #pragma unroll
    for (int th = 0; th < TILE_SIZE_H/2; th++) {
        #pragma unroll
        for (int tw = 0; tw < TILE_SIZE_W/2; tw++) {
            const int oh = tile_start_h + th * 2 + ty % 2;
            const int ow = tile_start_w + tw * 2 + tx % 2;

            if (oh < H_out && ow < W_out) {
                const int output_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
                output[output_idx] = partial_sums[th][tw];
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

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (W_out + TILE_SIZE_W - 1) / TILE_SIZE_W,
        (H_out + TILE_SIZE_H - 1) / TILE_SIZE_H,
        N * C_out
    );

    conv_transpose2d_forward_kernel_tiled<<<grid, block>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with tiled workload distribution",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}