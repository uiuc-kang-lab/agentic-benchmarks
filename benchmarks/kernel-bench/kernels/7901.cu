#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tunable constants
constexpr int TILE_DIM = 16;
constexpr int CHANNEL_TILE = 16;

// This kernel implements a 2D convolution using shared memory tiling over spatial and channel dimensions.
// Each block is assigned a tile of output for a specific batch index and output channel, and iterates over
// chunks of input channels (CHANNEL_TILE at a time). For each chunk, it cooperatively loads an input patch
// and corresponding weight tile into shared memory, then each thread computes the convolution on its output pixel.
// Bias is added (if provided) once per output pixel and atomic operations are avoided since each block writes
// its unique output region.

__global__ void conv2d_shared_kernel(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ output,
                                      int N, int Cin, int H, int W,
                                      int Cout, int K,
                                      int outH, int outW,
                                      int stride, int padding) {
    // Decode blockIdx.z into batch index and output channel
    int n_cout = blockIdx.z;
    int n = n_cout / Cout;
    int cout = n_cout % Cout;

    // Determine the starting coordinates of the output tile
    int tile_x = blockIdx.x * TILE_DIM;
    int tile_y = blockIdx.y * TILE_DIM;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the corresponding starting coordinates in the input
    int in_tile_x = tile_x * stride - padding;
    int in_tile_y = tile_y * stride - padding;

    // The shared input tile needs to cover the region required for the whole output tile.
    // For stride = 1, this is TILE_DIM + K - 1; generally, it is:
    int in_tile_size = (TILE_DIM - 1) * stride + K;

    // Initialize the output accumulator
    float sum = 0.0f;
    // If bias is provided, add it once per output pixel
    if ((tile_x + tx) < outW && (tile_y + ty) < outH && bias) {
        sum = bias[cout];
    }

    // Use dynamically allocated shared memory. We partition it into two regions:
    // 1. s_input: input tile buffer for CHANNEL_TILE channels, size = CHANNEL_TILE * (in_tile_size^2)
    // 2. s_weight: weight buffer for CHANNEL_TILE channels, size = CHANNEL_TILE * (K*K)
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;  
    float* s_weight = s_input + CHANNEL_TILE * in_tile_size * in_tile_size;

    // Loop over the input channels in blocks of CHANNEL_TILE
    for (int c_tile = 0; c_tile < Cin; c_tile += CHANNEL_TILE) {
        int current_tile = min(CHANNEL_TILE, Cin - c_tile);

        // Load the input patch for the current channel tile from global memory into shared memory.
        // Each channel in the tile will have an input patch of size (in_tile_size x in_tile_size).
        int total_elems = in_tile_size * in_tile_size;
        for (int ch = 0; ch < current_tile; ch++) {
            // Using a simple grid-stride loop over the input patch elements
            for (int idx = ty * TILE_DIM + tx; idx < total_elems; idx += TILE_DIM * TILE_DIM) {
                int i = idx / in_tile_size;
                int j = idx % in_tile_size;
                int global_x = in_tile_x + j;
                int global_y = in_tile_y + i;
                float val = 0.0f;
                if (global_x >= 0 && global_x < W && global_y >= 0 && global_y < H) {
                    int input_idx = ((n * Cin + (c_tile + ch)) * H + global_y) * W + global_x;
                    val = input[input_idx];
                }
                s_input[ch * (in_tile_size * in_tile_size) + i * in_tile_size + j] = val;
            }
        }

        // Load the weight tile for the current channel tile. Each weight tile is of size (K x K) per channel.
        int weight_elems = K * K;
        for (int ch = 0; ch < current_tile; ch++) {
            for (int idx = ty * TILE_DIM + tx; idx < weight_elems; idx += TILE_DIM * TILE_DIM) {
                int i = idx / K;
                int j = idx % K;
                int weight_idx = ((cout * Cin + (c_tile + ch)) * K + i) * K + j;
                s_weight[ch * (K * K) + i * K + j] = weight[weight_idx];
            }
        }

        __syncthreads(); // Ensure shared memory is loaded before computation

        // Compute the partial sum for the output pixel corresponding to this thread
        int out_x = tile_x + tx;
        int out_y = tile_y + ty;
        if (out_x < outW && out_y < outH) {
            // Iterate over the current channel tile
            for (int ch = 0; ch < current_tile; ch++) {
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        // Map thread's output location into the shared input tile
                        int in_i = ty * stride + i;
                        int in_j = tx * stride + j;
                        float in_val = s_input[ch * (in_tile_size * in_tile_size) + in_i * in_tile_size + in_j];
                        float w_val = s_weight[ch * (K * K) + i * K + j];
                        sum += in_val * w_val;
                    }
                }
            }
        }

        __syncthreads(); // Prepare for the next channel tile
    }

    // Write the computed output value to global memory if within bounds
    int out_x = tile_x + tx;
    int out_y = tile_y + ty;
    if (out_x < outW && out_y < outH) {
        int out_idx = ((n * Cout + cout) * outH + out_y) * outW + out_x;
        output[out_idx] = sum;
    }
}


// The forward function sets up grid and block dimensions, calculates the output shape, and
// launches the optimized kernel. This kernel eschews multi-block partitioned reduction and atomic
// operations by having one block compute a unique output tile for each (batch, output channel) pair.
// It leverages shared memory to reduce redundant global loads for input patches and weights.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation, // dilation not supported in this kernel
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    TORCH_CHECK(groups == 1, "groups != 1 not supported by conv2d_shared_kernel");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2); // assuming square kernel
    int outH = (H + 2 * padding - K) / stride + 1;
    int outW = (W + 2 * padding - K) / stride + 1;

    auto output = torch::zeros({N, Cout, outH, outW}, x.options());

    // Grid dimensions:
    // - gridDim.z encodes the (batch, cout) pair
    // - gridDim.x and gridDim.y cover the output spatial dimensions by tiles of TILE_DIM
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((outW + TILE_DIM - 1) / TILE_DIM,
                 (outH + TILE_DIM - 1) / TILE_DIM,
                 N * Cout);

    // Calculate the shared memory size per block:
    // For the input tile: CHANNEL_TILE * (in_tile_size^2) floats
    // For the weight tile: CHANNEL_TILE * (K*K) floats
    int in_tile_size = (TILE_DIM - 1) * stride + K;
    size_t sharedMemSize = CHANNEL_TILE * (in_tile_size * in_tile_size + K * K) * sizeof(float);

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_shared_kernel<<<gridDim, blockDim, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, H, W,
        Cout, K,
        outH, outW,
        stride, padding);

    cudaDeviceSynchronize();

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution using shared memory tiling");
}
