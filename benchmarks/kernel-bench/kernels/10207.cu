#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory for weights and bias
__constant__ float c_weight[1024 * 1024]; // Adjust size based on expected maximum
__constant__ float c_bias[1024];

__global__ void forward_kernel(
    const float* x,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W,
    bool has_bias
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    
    if (index >= total_elements) return;
    
    // Decompose linear index into 4D tensor coordinates
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    
    // Use constant memory for weights
    for (int ic = 0; ic < IC; ++ic) {
        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        const int w_offset = oc * IC + ic;
        sum += x[x_offset] * c_weight[w_offset];
    }
    
    // Use constant memory for bias
    output[index] = has_bias ? sum + c_bias[oc] : sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    
    // Copy weight to constant memory
    const size_t weight_size = OC * IC * sizeof(float);
    TORCH_CHECK(weight_size <= 1024 * 1024 * sizeof(float), "Weight tensor too large for constant memory");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_size);

    bool has_bias = bias.has_value();
    if (has_bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
        cudaMemcpyToSymbol(c_bias, bias->data_ptr<float>(), OC * sizeof(float));
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const int threads = 256;
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    
    forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, IC, OC, H, W,
        has_bias
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}