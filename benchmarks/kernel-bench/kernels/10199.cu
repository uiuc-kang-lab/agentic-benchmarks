#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for bias
__constant__ float c_bias[1024];

__device__ void calculate_indices(
    const int index,
    const int W,
    const int H,
    const int OC,
    int& w,
    int& h,
    int& oc,
    int& b
) {
    w = index % W;
    h = (index / W) % H;
    oc = (index / (W * H)) % OC;
    b = index / (W * H * OC);
}

__device__ float compute_conv(
    const float* x,
    const float* weight,
    const int b,
    const int oc,
    const int h,
    const int w,
    const int IC,
    const int H,
    const int W
) {
    float sum = 0.0f;
    #pragma unroll
    for (int ic = 0; ic < IC; ++ic) {
        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        const int w_offset = oc * IC + ic;
        sum += x[x_offset] * weight[w_offset];
    }
    return sum;
}

__global__ void forward_kernel(
    const float* x,
    const float* weight,
    const bool has_bias,
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
    
    int w, h, oc, b;
    calculate_indices(index, W, H, OC, w, h, oc, b);
    
    float result = compute_conv(x, weight, b, oc, h, w, IC, H, W);
    
    if (has_bias) {
        result += c_bias[oc];
    }
    
    output[index] = result;
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
    
    bool has_bias = bias.has_value();
    if (has_bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
        cudaMemcpyToSymbol(c_bias, bias->data_ptr<float>(), OC * sizeof(float));
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int threads = 256;
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    
    forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, has_bias, out_ptr,
        B, IC, OC, H, W
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}