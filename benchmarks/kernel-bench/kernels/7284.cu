#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16

__global__ void conv2d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;
    int c_out = bz;

    if (h_out >= H_out || w_out >= W_out) return;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    for (int n = 0; n < N; n++) {
        for (int c_in = c_in_start; c_in < c_in_end; c_in += TILE_SIZE) {
            // Load input tile
            if (c_in + tx < c_in_end && h_out < H_out && w_out < W_out) {
                int h_in = h_out * stride_h - padding_h;
                int w_in = w_out * stride_w - padding_w;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    shared_input[ty][tx] = input[((n * C_in + c_in + tx) * H_in + h_in) * W_in + w_in];
                } else {
                    shared_input[ty][tx] = 0.0f;
                }
            }

            // Load weight tile
            if (tx < K_w && ty < K_h && c_in + threadIdx.z < c_in_end) {
                shared_weight[ty][tx] = weight[(c_out * (C_in / groups) + (c_in - c_in_start) + threadIdx.z) * K_h * K_w + ty * K_w + tx];
            }

            __syncthreads();

            // Compute convolution for this tile
            for (int k = 0; k < min(TILE_SIZE, c_in_end - c_in); k++) {
                for (int p = 0; p < K_h; p++) {
                    for (int q = 0; q < K_w; q++) {
                        int h_in = h_out * stride_h - padding_h + p * dilation_h;
                        int w_in = w_out * stride_w - padding_w + q * dilation_w;
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            value += shared_input[ty][k] * shared_weight[p][q];
                        }
                    }
                }
            }

            __syncthreads();
        }
    }

    if (h_out < H_out && w_out < W_out) {
        for (int n = 0; n < N; n++) {
            output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = value;
        }
    }
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    int64_t H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    int64_t W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        C_out
    );

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().contiguous().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    conv2d_cuda_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Custom 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}