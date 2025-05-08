#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    // Each thread block handles a slice of the output
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Calculate which batch and output channel this block processes
    const int flat_block_idx = blockIdx.x;
    const int b = flat_block_idx / (OC * H);
    const int oc = (flat_block_idx / H) % OC;
    const int h = flat_block_idx % H;
    
    // Each thread processes elements along the width dimension
    for (int w = tid; w < W; w += block_size) {
        float sum = 0.0f;
        
        // Input base offset for this batch and spatial location
        const int x_base = b * IC * H * W + h * W + w;
        
        // Process all input channels
        #pragma unroll 4
        for (int ic = 0; ic < IC; ++ic) {
            sum += x[x_base + ic * H * W] * weight[oc * IC + ic];
        }
        
        // Write output with coalesced access pattern
        const int out_idx = b * OC * H * W + oc * H * W + h * W + w;
        output[out_idx] = bias ? sum + bias[oc] : sum;
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

    // Launch kernel with improved thread organization
    const int threads = 256;
    const int blocks = B * OC * H;
    
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