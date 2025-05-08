#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel variant when no bias is provided
__global__ void forward_kernel_no_bias(
    const float* x,
    const float* weight,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_elements = B * OC * H * W;
    
    for (int index = tid; index < total_elements; index += stride) {
        // Decompose linear index into 4D tensor coordinates
        int w = index % W;
        int h = (index / W) % H;
        int oc = (index / (W * H)) % OC;
        int b = index / (W * H * OC);
        
        float sum = 0.0f;
        // Perform 1x1 convolution (pointwise)
        for (int ic = 0; ic < IC; ++ic) {
            int x_offset = b * IC * H * W + ic * H * W + h * W + w;
            int weight_offset = oc * IC + ic; // Correctly calculate weight offset for 1x1 convolution
            sum += x[x_offset] * weight[weight_offset];
        }
        output[index] = sum;
    }
}

// Kernel variant when bias is provided
__global__ void forward_kernel_bias(
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_elements = B * OC * H * W;
    
    for (int index = tid; index < total_elements; index += stride) {
        // Decompose linear index into 4D tensor coordinates
        int w = index % W;
        int h = (index / W) % H;
        int oc = (index / (W * H)) % OC;
        int b = index / (W * H * OC);
        
        float sum = 0.0f;
        // Perform 1x1 convolution (pointwise)
        for (int ic = 0; ic < IC; ++ic) {
            int x_offset = b * IC * H * W + ic * H * W + h * W + w;
            int weight_offset = oc * IC + ic; // Correctly calculate weight offset for 1x1 convolution
            sum += x[x_offset] * weight[weight_offset];
        }
        // Uniform control flow: no conditional per thread
        output[index] = sum + bias[oc];
    }
}

// Host function dispatches the appropriate kernel based on bias presence
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
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == OC, "Bias/out channel mismatch");
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int threads = 256;
    int total_elements = B * OC * H * W;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = min(blocks, 65535);

    if (bias.has_value()) {
        const float* b_ptr = bias.value().data_ptr<float>();
        forward_kernel_bias<<<blocks, threads>>>(
            x_ptr, w_ptr, b_ptr, out_ptr,
            B, IC, OC, H, W
        );
    } else {
        forward_kernel_no_bias<<<blocks, threads>>>(
            x_ptr, w_ptr, out_ptr,
            B, IC, OC, H, W
        );
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}
