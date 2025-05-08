#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel without bias; using grid-stride loop for uniform control flow
__global__ void forward_kernel_nobias(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int IC, int OC, int H, int W
) {
    int total = B * OC * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to cover all output elements
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        // Decompose linear index into 4D tensor coordinates
        int w = idx % W;
        int h = (idx / W) % H;
        int oc = (idx / (W * H)) % OC;
        int b = idx / (W * H * OC);
        
        float sum = 0.0f;
        #pragma unroll
        for (int ic = 0; ic < IC; ++ic) {
            int x_offset = b * (IC * H * W) + ic * (H * W) + h * W + w;
            int w_offset = oc * IC + ic;
            sum += x[x_offset] * weight[w_offset];
        }
        output[idx] = sum;
    }
}

// Kernel with bias; bias addition is inlined without conditional branching per thread
__global__ void forward_kernel_bias(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int IC, int OC, int H, int W
) {
    int total = B * OC * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to ensure uniform work distribution
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        int w = idx % W;
        int h = (idx / W) % H;
        int oc = (idx / (W * H)) % OC;
        int b = idx / (W * H * OC);
        
        float sum = 0.0f;
        #pragma unroll
        for (int ic = 0; ic < IC; ++ic) {
            int x_offset = b * (IC * H * W) + ic * (H * W) + h * W + w;
            int w_offset = oc * IC + ic;
            sum += x[x_offset] * weight[w_offset];
        }
        // Uniform control flow for bias addition; no branch per thread
        output[idx] = sum + bias[oc];
    }
}

// CUDA forward function dispatching to the appropriate kernel variant
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
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    const int total_elements = B * OC * H * W;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    if (bias) {
        const float* bias_ptr = bias->data_ptr<float>();
        forward_kernel_bias<<<blocks, threads>>>(
            x_ptr, weight_ptr, bias_ptr, output_ptr,
            B, IC, OC, H, W
        );
    } else {
        forward_kernel_nobias<<<blocks, threads>>>(
            x_ptr, weight_ptr, output_ptr,
            B, IC, OC, H, W
        );
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA) with minimized warp divergence");
}
