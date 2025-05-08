#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int TILE_DIM = 16;
constexpr int CHANNEL_TILE = 16;

__global__ void optimized_conv2d_kernel(const float* __restrict__ input,
                                         const float* __restrict__ weight,
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         int N, int Cin, int H, int W,
                                         int Cout, int K,
                                         int outH, int outW,
                                         int stride, int padding, int dilation,
                                         int channel_tile, int partitions) {
    int partition_id = blockIdx.z % partitions;
    int n_cout = blockIdx.z / partitions;
    int n = n_cout / Cout;
    int cout = n_cout % Cout;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox < outW && oy < outH) {
        float partial = 0.0f;
        int cin_start = partition_id * channel_tile;
        int cin_end = min(cin_start + channel_tile, Cin);

        for (int cin = cin_start; cin < cin_end; ++cin) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    int in_y = oy * stride - padding + i * dilation;
                    int in_x = ox * stride - padding + j * dilation;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx = ((n * Cin + cin) * H + in_y) * W + in_x;
                        int weight_idx = ((cout * Cin + cin) * K + i) * K + j;
                        partial += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                    }
                }
            }
        }
        if (partition_id == 0 && bias != nullptr) {
            partial += __ldg(&bias[cout]);
        }

        int out_idx = ((n * Cout + cout) * outH + oy) * outW + ox;
        if (partitions > 1) {
            atomicAdd(&output[out_idx], partial);
        } else {
            output[out_idx] = partial;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    TORCH_CHECK(groups == 1, "groups != 1 not supported by this kernel");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2);
    int outH = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    int partitions = (Cin + CHANNEL_TILE - 1) / CHANNEL_TILE;
    auto output = torch::zeros({N, Cout, outH, outW}, x.options());

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((outW + TILE_DIM - 1) / TILE_DIM,
                 (outH + TILE_DIM - 1) / TILE_DIM,
                 N * Cout * partitions);

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    optimized_conv2d_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, H, W, Cout, K,
        outH, outW,
        stride, padding, dilation,
        CHANNEL_TILE, partitions);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution with atomic reduction and __ldg() use");
}