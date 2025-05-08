#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses grid-stride loops and a uniform bias flag to minimize warp divergence
__global__ void forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int B,
    int IC,
    int OC,
    int H,
    int W,
    int use_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop reduces divergence by having all threads perform similar iterations
    for (; idx < total_elements; idx += stride) {
        // Decompose the linear index into 4D tensor coordinates
        int w = idx % W;
        int h = (idx / W) % H;
        int oc = (idx / (W * H)) % OC;
        int b = idx / (W * H * OC);

        float sum = 0.0f;
        // Perform 1x1 convolution equivalent (inner product over input channels)
        for (int ic = 0; ic < IC; ++ic) {
            int x_offset = b * IC * H * W + ic * H * W + h * W + w;
            int w_offset = oc * IC + ic;
            sum += x[x_offset] * weight[w_offset];
        }
        // Use the uniform bias flag to add bias without branching
        output[idx] = sum + use_bias * bias[oc];
    }
}

// Host function
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");

    int use_bias = 0;
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = *bias_opt;
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias.size(0) == weight.size(0), "Bias/out channel mismatch");
        use_bias = 1;
        bias_ptr = bias.data_ptr<float>();
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);
    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int total_elements = B * OC * H * W;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, bias_ptr, out_ptr, total_elements,
        B, IC, OC, H, W, use_bias
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Efficient pointwise 2D convolution forward (CUDA)");
}
