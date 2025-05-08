#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel implements 2D convolution for square input and square kernel with the explicit aim
// to align global memory accesses for coalescing. Threads in the same warp process adjacent
// output columns, ensuring that writes (and many reads) hit consecutive memory addresses.
// Only groups==1 is supported. Dilation is incorporated for correctness, and bias is optional.

__global__ void conv2d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int N,
                              int Cin,
                              int H,
                              int W,
                              int Cout,
                              int K,
                              int stride,
                              int padding,
                              int dilation,
                              int outH,
                              int outW) {
    // Compute the starting output coordinates for this block
    int block_output_x = blockIdx.x * blockDim.x;
    int block_output_y = blockIdx.y * blockDim.y;
    
    // Use gridDim.z to cover both batch and output channel dimensions
    int linear_idx = blockIdx.z; // linear index combining batch and output channel
    int n = linear_idx / Cout;
    int cout = linear_idx % Cout;

    // Each thread computes one output element
    int ox = block_output_x + threadIdx.x;
    int oy = block_output_y + threadIdx.y;

    if (ox < outW && oy < outH) {
        float sum = 0.0f;
        
        // Loop over input channels and kernel elements
        for (int cin = 0; cin < Cin; cin++) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    // Compute the corresponding input pixel location with dilation
                    int in_y = oy * stride - padding + i * dilation;
                    int in_x = ox * stride - padding + j * dilation;
                    
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx = ((n * Cin + cin) * H + in_y) * W + in_x;
                        int weight_idx = ((cout * Cin + cin) * K + i) * K + j;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[cout];
        }
        
        int output_idx = ((n * Cout + cout) * outH + oy) * outW + ox;
        output[output_idx] = sum;
    }
}

// The forward function sets up the kernel launch parameters ensuring that threads within a warp
// write to consecutive memory locations in the output tensor, achieving memory coalescing.
// Note: For simplicity, only groups == 1 is supported in this implementation.

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

    TORCH_CHECK(groups == 1, "groups != 1 not supported by custom kernel");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2); // Square kernel assumed (K x K)

    // Compute the output dimensions taking dilation into account
    int outH = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Define block dimensions to promote memory coalescing along the output width dimension.
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((outW + TILE_WIDTH - 1) / TILE_WIDTH,
                 (outH + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 N * Cout);

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, H, W, Cout, K,
        stride, padding, dilation, outH, outW);

    cudaDeviceSynchronize();
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with improved memory coalescing");
}
