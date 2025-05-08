#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Tile sizes for shared memory
#define TILE_SIZE 16
#define KERNEL_TILE 3

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
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[KERNEL_TILE][KERNEL_TILE];

    // Block and thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate base indices
    const int n = bz / C_out;
    const int oc = bz % C_out;
    const int out_h_base = by * TILE_SIZE;
    const int out_w_base = bx * TILE_SIZE;

    // Output coordinates for this thread
    const int oh = out_h_base + ty;
    const int ow = out_w_base + tx;

    // Early exit if outside output bounds
    if (oh >= H_out || ow >= W_out) return;

    float sum = 0.0f;

    // Process input channels in tiles
    for (int ic = 0; ic < C_in; ++ic) {
        // Process kernel in tiles
        for (int kh_start = 0; kh_start < kH; kh_start += KERNEL_TILE) {
            for (int kw_start = 0; kw_start < kW; kw_start += KERNEL_TILE) {
                
                // Load kernel weights into shared memory
                if (tx < KERNEL_TILE && ty < KERNEL_TILE) {
                    int kh = kh_start + ty;
                    int kw = kw_start + tx;
                    if (kh < kH && kw < kW) {
                        shared_weight[ty][tx] = weight[((ic * C_out + oc) * kH + kh) * kW + kw];
                    } else {
                        shared_weight[ty][tx] = 0.0f;
                    }
                }
                __syncthreads();

                // Process this tile of the kernel
                #pragma unroll
                for (int k_tile_h = 0; k_tile_h < KERNEL_TILE; ++k_tile_h) {
                    #pragma unroll
                    for (int k_tile_w = 0; k_tile_w < KERNEL_TILE; ++k_tile_w) {
                        int kh = kh_start + k_tile_h;
                        int kw = kw_start + k_tile_w;
                        
                        if (kh < kH && kw < kW) {
                            int i_val = oh + pH - kh;
                            int j_val = ow + pW - kw;

                            if ((i_val % sH == 0) && (j_val % sW == 0)) {
                                int i_in = i_val / sH;
                                int j_in = j_val / sW;

                                if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                                    float in_val = input[((n * C_in + ic) * H_in + i_in) * W_in + j_in];
                                    sum += in_val * shared_weight[k_tile_h][k_tile_w];
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write output
    if (oh < H_out && ow < W_out) {
        output[((n * C_out + oc) * H_out + oh) * W_out + ow] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
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

    // Calculate grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        N * C_out
    );

    conv_transpose2d_forward_kernel<<<grid, block>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with tiled shared memory",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}