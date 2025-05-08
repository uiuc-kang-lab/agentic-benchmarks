#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a 3D grid to directly map the problem domain: the x and y dimensions of the block map
// to the spatial dimensions of the output, and the z-dimension of the grid maps to the combined batch and
// output channel dimensions. This ensures that each thread is responsible for exactly one output element,
// minimizing divergence and maximizing memory coalescing. The kernel computes 2D convolution with optional bias,
// including support for dilation and padding. Only groups == 1 is supported.

__global__ void conv2d_kernel(
    const float* __restrict__ input,
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

    // Map thread indices to output spatial coordinates
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    // Map gridDim.z to combined (batch, out_channel) index
    int linearIndex = blockIdx.z; // ranges from 0 to N * Cout - 1
    int n = linearIndex / Cout;
    int oc = linearIndex % Cout;

    if (ox < outW && oy < outH) {
        float sum = 0.0f;
        // Loop over input channels and the kernel window
        for (int c = 0; c < Cin; ++c) {
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int in_y = oy * stride - padding + i * dilation;
                    int in_x = ox * stride - padding + j * dilation;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx = ((n * Cin + c) * H + in_y) * W + in_x;
                        int weight_idx = ((oc * Cin + c) * K + i) * K + j;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = ((n * Cout + oc) * outH + oy) * outW + ox;
        output[output_idx] = sum;
    }
}

// The forward function calculates output dimensions and launches the kernel using a 3D grid, where:
//   gridDim.x covers the output width,
//   gridDim.y covers the output height,
//   gridDim.z covers the batch and output channels combined.

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
    int K = weight.size(2); // assuming square kernel (K x K)

    // Compute output dimensions with dilation taken into account
    int outH = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Define a 2D block that maps to the output spatial dimensions
    dim3 blockDim(16, 16);
    // Grid dimensions: x covers columns, y covers rows, z covers batch * output channels
    dim3 gridDim(
        (outW + blockDim.x - 1) / blockDim.x,
        (outH + blockDim.y - 1) / blockDim.y,
        N * Cout
    );

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &forward, "CUDA forward function for 2D convolution with 3D thread mapping");
}
