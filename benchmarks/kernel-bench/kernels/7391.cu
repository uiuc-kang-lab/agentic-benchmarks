#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32  // Size of output tile per block
#define BLOCK_SIZE 256  // Number of threads per block

__global__ void conv2d_cuda_kernel_tiled(
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
    // Calculate tile indices
    const int tile_h = blockIdx.y * TILE_SIZE;
    const int tile_w = blockIdx.z * TILE_SIZE;
    const int batch_feature = blockIdx.x;
    const int n = batch_feature / C_out;
    const int c_out = batch_feature % C_out;

    // Calculate thread position within tile
    const int thread_idx = threadIdx.x;
    const int tile_row = thread_idx / TILE_SIZE;
    const int tile_col = thread_idx % TILE_SIZE;

    // Calculate output position
    const int h_out = tile_h + tile_row;
    const int w_out = tile_w + tile_col;

    // Early exit if outside output bounds
    if (h_out >= H_out || w_out >= W_out || n >= N) return;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    
    #pragma unroll 4
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        #pragma unroll 2
        for (int k_h = 0; k_h < K_h; ++k_h) {
            const int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            if ((unsigned)h_in >= (unsigned)H_in) continue;

            #pragma unroll 2
            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                if ((unsigned)w_in >= (unsigned)W_in) continue;

                const int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                const int weight_idx = (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
                
                value += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Write output
    const int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = value;
}

torch::Tensor conv2d_cuda_tiled(
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

    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    const auto C_out = weight.size(0);
    const auto K_h = weight.size(2);
    const auto K_w = weight.size(3);

    const auto H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    const auto W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Calculate grid dimensions for tiled approach
    dim3 blocks(
        N * C_out,  // Batch and output features
        (H_out + TILE_SIZE - 1) / TILE_SIZE,  // Height tiles
        (W_out + TILE_SIZE - 1) / TILE_SIZE   // Width tiles
    );

    const int threads = BLOCK_SIZE;

    conv2d_cuda_kernel_tiled<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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
    m.def("forward", &conv2d_cuda_tiled, "Tiled 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}