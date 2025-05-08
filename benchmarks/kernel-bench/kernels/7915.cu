/*
Combined CUDA kernel for 2D convolution using both input and weight tiling into shared memory.
Each block computes a tile of output pixels for a given (batch, out_channel) pair.
The kernel loads the full weight filter for the output channel into dynamic shared memory once per block
and then, for each input channel, loads the needed input tile into shared memory to reduce redundant
global memory accesses.
Note: This example assumes square kernels and supports a general stride (though dilation and groups are not supported).
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Macro to check if tensor is CUDA and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Combined kernel: uses shared memory tiling for both weights and input patches
// Parameters:
// input:  (N, Cin, H, W)
// weight: (Cout, Cin, K, K) 
// output: (N, Cout, outH, outW)
// K is the kernel size, stride and padding are as given

// Grid mapping: each block is assigned to a tile of output pixels for a given (n, oc).
// Block dimensions: (BLOCK_SIZE, BLOCK_SIZE).
// Grid z-dimension: N * Cout.
// Dynamic shared memory layout: first part for weights (Cin*K*K floats), then input tile (shared_width * shared_height floats).

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N,
    const int Cin,
    const int H,
    const int W,
    const int Cout,
    const int K,
    const int outH,
    const int outW,
    const int stride,
    const int padding) {

    // Decode blockIdx.z into batch index and output channel
    int n = blockIdx.z / Cout;
    int oc = blockIdx.z % Cout;

    // Compute output tile start indices
    int ox0 = blockIdx.x * BLOCK_SIZE;
    int oy0 = blockIdx.y * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ox = ox0 + tx; // output x coordinate
    int oy = oy0 + ty; // output y coordinate

    // Compute shared tile dimensions for the input patch.
    // The tile covers the input region needed by the output tile computed by this block.
    // For an output tile of size (BLOCK_SIZE x BLOCK_SIZE), the input tile size is:
    // shared_width  = BLOCK_SIZE * stride + (K - stride)
    // shared_height = BLOCK_SIZE * stride + (K - stride)
    int shared_width = BLOCK_SIZE * stride + (K - stride);
    int shared_height = BLOCK_SIZE * stride + (K - stride);

    // Allocate dynamic shared memory: first area for weights (fixed size) then input tile
    extern __shared__ float shared_mem[];
    // Weight shared memory for this block: size = Cin * K * K
    float* s_weight = shared_mem;              
    // Input tile: size = shared_width * shared_height
    float* s_input = s_weight + (Cin * K * K);

    // Each block is for fixed out_channel, so load weights for this output channel
    int total_weight = Cin * K * K;
    int tid = ty * BLOCK_SIZE + tx; // linear thread id within block
    for (int i = tid; i < total_weight; i += BLOCK_SIZE * BLOCK_SIZE) {
        s_weight[i] = weight[((oc * Cin) * K * K) + i];
    }
    __syncthreads();

    float sum = 0.0f;

    // Loop over all input channels
    for (int c = 0; c < Cin; c++) {
        // Load the input tile for channel c into shared memory
        // The top-left corner of the input tile for this block is computed from the output tile start
        int total_shared = shared_width * shared_height;
        for (int idx = tid; idx < total_shared; idx += BLOCK_SIZE * BLOCK_SIZE) {
            int local_y = idx / shared_width;
            int local_x = idx % shared_width;
            // Compute global input coordinate
            int in_y = oy0 * stride - padding + local_y;
            int in_x = ox0 * stride - padding + local_x;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                s_input[idx] = input[((n * Cin + c) * H + in_y) * W + in_x];
            } else {
                s_input[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Only threads that correspond to a valid output location compute the convolution
        if (ox < outW && oy < outH) {
            // The top-left index in the shared input for this output element is:
            int local_in_y = ty * stride;  // equivalent to (oy - oy0) * stride
            int local_in_x = tx * stride;  // equivalent to (ox - ox0) * stride

            int weight_offset = c * (K * K);
            // Loop over the kernel window
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    float in_val = s_input[(local_in_y + i) * shared_width + (local_in_x + j)];
                    float w_val = s_weight[weight_offset + i * K + j];
                    sum += in_val * w_val;
                }
            }
        }
        __syncthreads(); // Ensure shared input is not overwritten before all threads are done
    }

    // Write the computed output value
    if (ox < outW && oy < outH) {
        output[((n * Cout + oc) * outH + oy) * outW + ox] = sum;
    }
}


// Host function to launch the convolution kernel
// Assumes input has shape [N, Cin, H, W] and weight has shape [Cout, Cin, K, K]
// Bias is optional. Dilation and groups are not supported in this kernel.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,  // Not applied
    int groups) {  // Only groups == 1 is supported

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    TORCH_CHECK(groups == 1, "groups != 1 is not supported by this kernel");
    TORCH_CHECK(dilation == 1, "dilation != 1 is not supported by this kernel");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2); // assuming square kernel

    int outH = (H + 2 * padding - K) / stride + 1;
    int outW = (W + 2 * padding - K) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((outW + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (outH + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 N * Cout);

    // Compute shared memory size:
    // We need space for weights: (Cin * K * K) floats
    // and for the input tile: shared_width * shared_height floats, where:
    int shared_width = BLOCK_SIZE * stride + (K - stride);
    int shared_height = BLOCK_SIZE * stride + (K - stride);
    size_t shared_mem_size = sizeof(float) * (Cin * K * K + shared_width * shared_height);

    conv2d_tiled_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, H, W, Cout, K, outH, outW, stride, padding);

    cudaDeviceSynchronize();

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution combining weight and input tiling");
}
