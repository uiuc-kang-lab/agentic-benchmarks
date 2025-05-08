#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    // Calculate 2D position within the HxW grid
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (w >= W || h >= H) return;
    
    // Process all batches and output channels for this H,W position
    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = 0.0f;
            
            // Compute convolution for this output position
            for (int ic = 0; ic < IC; ++ic) {
                const int x_idx = ((b * IC + ic) * H + h) * W + w;
                sum += x[x_idx] * weight[oc * IC + ic];
            }
            
            // Write output with coalesced access pattern
            const int out_idx = ((b * OC + oc) * H + h) * W + w;
            output[out_idx] = (b_ptr != nullptr) ? sum + b_ptr[oc] : sum;
        }
    }
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

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Launch kernel with 2D grid and block configuration
    dim3 threads(16, 16);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y
    );
    
    forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}