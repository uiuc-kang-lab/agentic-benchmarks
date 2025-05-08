#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define a tunable block size. Experiment with values 32, 64, 128, 256, 512.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    // Decompose linear index into tensor coordinates
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    // Perform 1x1 convolution over input channels
    for (int ic = 0; ic < IC; ++ic) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        int w_offset = oc * IC + ic;
        sum += x[x_offset] * weight[w_offset];
    }

    output[index] = bias ? (sum + bias[oc]) : sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) {
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Use the tunable BLOCK_SIZE for threads per block
    int threads = BLOCK_SIZE;
    int total_elements = B * OC * H * W;
    int blocks = (total_elements + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(x_ptr, w_ptr, b_ptr, out_ptr, B, IC, OC, H, W);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized Pointwise 2D Convolution forward (CUDA)");
}
