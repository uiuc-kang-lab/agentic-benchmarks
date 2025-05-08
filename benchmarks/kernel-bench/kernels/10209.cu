#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each warp computes one output element using warp-level reduction
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
    // Each warp (32 threads) computes one output element
    const unsigned int warpSize = 32;
    // Global warp index
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    int total_elements = B * OC * H * W;
    if (warpId >= total_elements) return;

    // Decompose warpId into 4D indices: b, oc, h, w
    int w_idx = warpId % W;
    int tmp = warpId / W;
    int h_idx = tmp % H;
    tmp = tmp / H;
    int oc = tmp % OC;
    int b = tmp / OC;

    // Each thread in the warp computes a partial sum over a subset of IC channels
    float partial_sum = 0.0f;
    for (int ic = lane; ic < IC; ic += warpSize) {
        int x_offset = b * IC * H * W + ic * H * W + h_idx * W + w_idx;
        int w_offset = oc * IC + ic;  // weight laid out as [OC, IC, 1, 1]
        partial_sum += x[x_offset] * weight[w_offset];
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        output[warpId] = (bias != nullptr) ? (partial_sum + bias[oc]) : partial_sum;
    }
}


// CUDA forward function
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    // Input validation
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
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

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Each warp (32 threads) computes one output element
    int total_elements = B * OC * H * W;
    const unsigned int warpSize = 32;
    int totalThreads = total_elements * warpSize;
    int threadsPerBlock = 256;  // multiple of warpSize
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    forward_kernel<<<blocks, threadsPerBlock>>>(
         x.data_ptr<float>(),
         weight.data_ptr<float>(),
         bias ? bias->data_ptr<float>() : nullptr,
         output.data_ptr<float>(),
         B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward with warp-level reduction (CUDA)");
}
