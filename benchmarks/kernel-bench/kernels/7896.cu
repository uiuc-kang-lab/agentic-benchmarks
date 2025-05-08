#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a shared-memory tiling approach to reuse input data across threads
// within the same output tile, eliminating the need for global atomic operations.
// Each block computes a tile of the output for a given (n, oc). For each input channel,
// the required input tile is loaded into shared memory and used by all threads in the block to
// compute a partial sum. The partial sums are accumulated in registers and then written to global
// memory, ensuring correctness while minimizing atomic operations.

__global__ void conv2d_shared_tile_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Cin, int H, int W,
    int Cout, int K,
    int outH, int outW,
    int stride, int padding, int dilation) {

    // Assume square block: TS x TS
    const int TS = blockDim.x;  // blockDim.x == blockDim.y
    // Compute dimensions of the shared memory tile for the current input channel.
    int tile_in_width  = (TS - 1) * stride + (K - 1) * dilation + 1;
    int tile_in_height = (TS - 1) * stride + (K - 1) * dilation + 1;

    // Determine the top-left output coordinate for the block.
    int out_x0 = blockIdx.x * TS;
    int out_y0 = blockIdx.y * TS;

    // Decode batch index and output channel from blockIdx.z: gridDim.z = N * Cout
    int n  = blockIdx.z / Cout;
    int oc = blockIdx.z % Cout;

    // Each thread computes one output element within the tile.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = out_x0 + tx;
    int out_y = out_y0 + ty;

    float sum = 0.0f;

    extern __shared__ float shmem[];  // Shared memory for one input channel tile

    // Loop over input channels
    for (int ic = 0; ic < Cin; ic++) {
        // Pointer to the current input channel for sample n
        const float* input_ptr = input + ((n * Cin + ic) * H * W);

        // Compute the top-left coordinate in the input for the tile
        int in_x0 = out_x0 * stride - padding;
        int in_y0 = out_y0 * stride - padding;

        // Load the input tile into shared memory.
        int tile_size = tile_in_width * tile_in_height;
        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        int block_threads = blockDim.x * blockDim.y;
        
        for (int idx = thread_id; idx < tile_size; idx += block_threads) {
            int r = idx / tile_in_width;
            int c = idx % tile_in_width;
            int in_x = in_x0 + c;
            int in_y = in_y0 + r;
            float val = 0.0f;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                val = input_ptr[in_y * W + in_x];
            }
            shmem[idx] = val;
        }
        __syncthreads();

        // Compute convolution for this input channel using shared memory
        if (out_x < outW && out_y < outH) {
            for (int kr = 0; kr < K; kr++) {
                for (int kc = 0; kc < K; kc++) {
                    // Calculate the corresponding position in shared memory
                    int smem_row = ty * stride + kr * dilation;
                    int smem_col = tx * stride + kc * dilation;
                    float in_val = shmem[smem_row * tile_in_width + smem_col];
                    // Weight is stored as [Cout, Cin, K, K]
                    float w = weight[ ((oc * Cin + ic) * K + kr) * K + kc ];
                    sum += in_val * w;
                }
            }
        }
        __syncthreads();  // Ensure shared memory is ready for next input channel
    }

    // Write the computed output element if within bounds, adding bias if available
    if (out_x < outW && out_y < outH) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_idx = ((n * Cout + oc) * outH + out_y) * outW + out_x;
        output[out_idx] = sum;
    }
}


// Forward function that sets up kernel launch parameters and avoids unnecessary atomic operations
// by ensuring each output element is computed exclusively by one thread in a block.

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
    TORCH_CHECK(groups == 1, "Only groups == 1 is supported");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2);  // assuming square kernel

    // Compute output dimensions taking dilation into account
    int outH = (H + 2 * padding - (K - 1) * dilation - 1) / stride + 1;
    int outW = (W + 2 * padding - (K - 1) * dilation - 1) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Choose tile size for the output (e.g., 16x16)
    int TS = 16;
    dim3 blockDim(TS, TS);
    dim3 gridDim((outW + TS - 1) / TS, (outH + TS - 1) / TS, N * Cout);

    // Compute shared memory size per block
    int tile_in_width  = (TS - 1) * stride + (K - 1) * dilation + 1;
    int tile_in_height = (TS - 1) * stride + (K - 1) * dilation + 1;
    size_t shmem_size = tile_in_width * tile_in_height * sizeof(float);

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_shared_tile_kernel<<<gridDim, blockDim, shmem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, H, W,
        Cout, K,
        outH, outW,
        stride, padding, dilation);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using shared memory tiling without unnecessary atomics");
}
