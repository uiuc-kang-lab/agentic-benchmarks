#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile dimensions for output spatial blocking
constexpr int TILE_DIM = 16;
// Partition size for input channels to enable reduction across blocks
constexpr int CHANNEL_TILE = 16;

// This kernel partitions the convolution reduction over the input channel dimension.
// Each block processes a subset of input channels (a partition) for a particular output element,
// and the partial sums are combined using atomicAdd only when more than one partition is present.
// The bias is added only by the first partition (partition_id == 0).
__global__ void conv2d_atomic_kernel(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ output,
                                      int N, int Cin, int H, int W,
                                      int Cout, int K,
                                      int outH, int outW,
                                      int stride, int padding,
                                      int channel_tile, int partitions) {
    // Decode blockIdx.z into a triplet: batch index (n), output channel (cout), and partition id
    int partition_id = blockIdx.z % partitions;
    int n_cout = blockIdx.z / partitions;  // n_cout in [0, N * Cout)
    int n = n_cout / Cout;
    int cout = n_cout % Cout;

    // Determine output spatial coordinates
    int ox = blockIdx.x * TILE_DIM + threadIdx.x;
    int oy = blockIdx.y * TILE_DIM + threadIdx.y;

    if (ox < outW && oy < outH) {
        float partial = 0.0f;
        int cin_start = partition_id * channel_tile;
        int cin_end = (cin_start + channel_tile < Cin) ? (cin_start + channel_tile) : Cin;

        // Accumulate over the partitioned subset of input channels and kernel window
        for (int cin = cin_start; cin < cin_end; ++cin) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    int in_y = oy * stride - padding + i;
                    int in_x = ox * stride - padding + j;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx = ((n * Cin + cin) * H + in_y) * W + in_x;
                        int weight_idx = ((cout * Cin + cin) * K + i) * K + j;
                        partial += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        // Only the first partition adds the bias
        if (partition_id == 0 && bias != nullptr) {
            partial += bias[cout];
        }

        int out_idx = ((n * Cout + cout) * outH + oy) * outW + ox;
        // If there are multiple partitions, use atomicAdd to combine partial sums
        if (partitions > 1) {
            atomicAdd(&output[out_idx], partial);
        } else {
            output[out_idx] = partial;
        }
    }
}

// The forward function partitions the reduction over input channels, launching multiple blocks
// per (n, cout) output element if necessary. Atomic operations are used only when more than
// one partition contributes to the same output element, thereby minimizing atomic contention.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation, // dilation is not used in this implementation
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
    int K = weight.size(2); // assuming square kernel
    int outH = (H + 2 * padding - K) / stride + 1;
    int outW = (W + 2 * padding - K) / stride + 1;

    // Determine the number of partitions over the input channel dimension
    int partitions = (Cin + CHANNEL_TILE - 1) / CHANNEL_TILE;

    auto output = torch::zeros({N, Cout, outH, outW}, x.options());

    dim3 blockDim(TILE_DIM, TILE_DIM);
    // gridDim.z encodes (n, cout, partition) to allow multiple blocks contributing to the same output element
    dim3 gridDim((outW + TILE_DIM - 1) / TILE_DIM,
                 (outH + TILE_DIM - 1) / TILE_DIM,
                 N * Cout * partitions);

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_atomic_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, H, W,
        Cout, K,
        outH, outW,
        stride, padding,
        CHANNEL_TILE, partitions);

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with atomic reduction on input channel partitions");
}
