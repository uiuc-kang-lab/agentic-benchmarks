#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define BLOCK_SIZE macro to experiment with different block sizes (32, 64, 128, 256, 512)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    // Convert linear index to 4D coordinates: b, oc, h, w
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        int w_offset = oc * IC + ic;
        sum += x[x_offset] * weight[w_offset];
    }
    
    // Apply bias if provided
    output[index] = bias ? sum + bias[oc] : sum;
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

    int B = x.size(0);
    int IC = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int OC = weight.size(0);

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

    int total_elements = B * OC * H * W;
    // Use tuned block size defined by BLOCK_SIZE. Experimentation suggests optimal performance on NVIDIA H100 with different block sizes.
    int threads = BLOCK_SIZE;
    int blocks = (total_elements + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized pointwise 2D convolution forward (CUDA)");
}
