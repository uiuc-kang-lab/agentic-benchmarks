#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that avoids divergent branching by assuming a valid bias pointer always exists
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = B * OC * H * W;
    for (int index = tid; index < total; index += stride) {
        // Decompose the linear index into 4D tensor coordinates (B, OC, H, W)
        int w = index % W;
        int h = (index / W) % H;
        int oc = (index / (W * H)) % OC;
        int b = index / (W * H * OC);

        float sum = 0.0f;
        // Unroll the inner loop if possible to reduce overhead
        #pragma unroll
        for (int ic = 0; ic < IC; ++ic) {
            int x_index = b * IC * H * W + ic * H * W + h * W + w;
            int w_index = oc * (IC * 1 * 1) + ic * (1 * 1) + 0 * 1 + 0;  // Proper 4D indexing for weight
            if (x_index < B * IC * H * W && w_index < OC * IC) {  // Bounds checking
                sum += x[x_index] * weight[w_index];
            }
        }
        // Uniform control flow: always add bias[oc] because bias is guaranteed to be valid
        output[index] = sum + bias[oc];
    }
}

// Forward function: ensures bias is always a valid pointer to avoid divergence
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor (OC, IC, 1, 1)");

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch in weight");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");

    // If bias is not provided, create a zero-filled tensor so that the kernel always adds bias[oc]
    torch::Tensor bias_tensor;
    if (!bias.has_value()) {
        bias_tensor = torch::zeros({OC}, weight.options());
        bias = bias_tensor;
    } else {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias->data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 256;
    int total_elements = B * OC * H * W;
    int blocks = (total_elements + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(x_ptr, weight_ptr, bias_ptr, output_ptr, B, IC, OC, H, W);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}
