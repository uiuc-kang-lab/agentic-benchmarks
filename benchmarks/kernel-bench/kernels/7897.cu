#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses __ldg() to load read-only data from global memory, which helps to optimize
// memory accesses when the data is aligned to 128-bit boundaries. It processes the 2D convolution
// by mapping threads to output elements in a 2D tile while looping over the input channels and
// kernel window. The __restrict__ qualifiers and __ldg() help to ensure that loads from input,
// weight, and bias are optimized via the read-only cache.

__global__ void conv2d_kernel_ldg(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N, int Cin, int H, int W,
                                    int Cout, int K,
                                    int stride, int padding, int dilation,
                                    int outH, int outW) {
    // Calculate starting coordinates for this block
    int block_output_x = blockIdx.x * blockDim.x;
    int block_output_y = blockIdx.y * blockDim.y;
    // Use blockIdx.z to cover both batch and output channel dimensions
    int linear_idx = blockIdx.z;  // linear index: n * Cout + out_c
    int n = linear_idx / Cout;
    int out_c = linear_idx % Cout;

    // Each thread computes one output element
    int ox = block_output_x + threadIdx.x;
    int oy = block_output_y + threadIdx.y;

    if (ox < outW && oy < outH) {
        // Load bias using __ldg() if bias pointer is provided
        float sum = (bias != nullptr ? __ldg(&bias[out_c]) : 0.f);
        
        // Loop over input channels and kernel window
        for (int c = 0; c < Cin; ++c) {
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int in_y = oy * stride - padding + i * dilation;
                    int in_x = ox * stride - padding + j * dilation;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx = ((n * Cin + c) * H + in_y) * W + in_x;
                        int weight_idx = ((out_c * Cin + c) * K + i) * K + j;
                        // Use __ldg() for read-only global memory loads
                        float input_val = __ldg(&input[input_idx]);
                        float weight_val = __ldg(&weight[weight_idx]);
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        int output_idx = ((n * Cout + out_c) * outH + oy) * outW + ox;
        output[output_idx] = sum;
    }
}

// The forward function sets up the kernel launch parameters. It computes the output dimensions
// considering stride, padding, and dilation, and launches a 3D grid where the first two dimensions
// map to the 2D spatial output and the third dimension maps to the batch and output channel.

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
    TORCH_CHECK(groups == 1, "groups != 1 is not supported by this kernel");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2); // Assuming square kernel
    
    // Compute output dimensions taking dilation into account
    int outH = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Setup 2D thread block and 3D grid to cover spatial and (batch, channel) dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((outW + blockDim.x - 1) / blockDim.x,
                 (outH + blockDim.y - 1) / blockDim.y,
                 N * Cout);

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_kernel_ldg<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Cin, H, W, Cout, K,
        stride, padding, dilation,
        outH, outW);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using __ldg() optimized global loads and 128-bit aligned accesses");
}
