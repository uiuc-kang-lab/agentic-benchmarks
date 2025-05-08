#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_WEIGHT_SIZE 16384

// Store weight data in constant memory for faster access
__constant__ float constant_weight[MAX_WEIGHT_SIZE];

__global__ void forward_kernel_const(
    const float* x,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    // Decompose linear index into 4D tensor coordinates (B, OC, H, W)
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    // 1x1 convolution: sum over input channels
    for (int ic = 0; ic < IC; ++ic) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        int w_offset = oc * IC + ic; // Index into constant weight memory
        sum += x[x_offset] * constant_weight[w_offset];
    }

    // Add bias if provided
    output[index] = bias ? sum + bias[oc] : sum;
}


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

    // Ensure the weight tensor fits in constant memory
    int weight_elems = OC * IC;
    TORCH_CHECK(weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory capacity");

    // Copy weight data to constant memory
    cudaError_t err = cudaMemcpyToSymbol(constant_weight, weight.data_ptr<float>(), weight_elems * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in cudaMemcpyToSymbol: ", cudaGetErrorString(err));

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw device pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Launch kernel
    const int threads = 256;
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    forward_kernel_const<<<blocks, threads>>>(
        x_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    // Check for errors
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward using constant memory (CUDA)");
}
