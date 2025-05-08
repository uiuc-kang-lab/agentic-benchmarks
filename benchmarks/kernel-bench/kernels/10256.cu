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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = B * OC * H * W;
    
    // Process elements with stride, aligned to warp size
    for (int index = tid; index < total_elements; index += stride) {
        // Decompose linear index into 4D tensor coordinates
        const int w = index % W;
        const int h = (index / W) % H;
        const int oc = (index / (W * H)) % OC;
        const int b = index / (W * H * OC);

        float sum = 0.0f;
        
        // Main computation loop without conditionals
        #pragma unroll 4
        for (int ic = 0; ic < IC; ++ic) {
            // Calculate correct offsets considering 4D weight tensor (OC, IC, 1, 1)
            const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
            const int w_offset = oc * IC * 1 * 1 + ic * 1 * 1 + 0 * 1 + 0;  // Access 4D weight tensor
            
            // Ensure we're within bounds
            if (b < B && oc < OC && h < H && w < W && ic < IC) {
                sum += x[x_offset] * weight[w_offset];
            }
        }
        
        // Bias addition moved outside the main loop
        output[index] = sum + (bias ? bias[oc] : 0.0f);
    }
}

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
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Align block size to warp size (32)
    const int threads = 256; // 8 warps per block
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    
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