#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using __ldg for read-only memory accesses and loop unrolling to encourage 128-bit aligned loads
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
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    // Decompose linear index into 4D coordinates
    int w_pos = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    
    // Loop unrolling by 4 to encourage aligned 128-bit loads for the weight tensor
    int ic_unroll = (IC / 4) * 4;
    for (int ic = 0; ic < ic_unroll; ic += 4) {
        int base_x = b * IC * H * W + h * W + w_pos;
        int base_w = oc * IC + ic;

        // Calculate offsets for 4 reads from x and weight
        int x_off0 = base_x + (ic + 0) * H * W;
        int x_off1 = base_x + (ic + 1) * H * W;
        int x_off2 = base_x + (ic + 2) * H * W;
        int x_off3 = base_x + (ic + 3) * H * W;

        int w_off0 = base_w + 0;
        int w_off1 = base_w + 1;
        int w_off2 = base_w + 2;
        int w_off3 = base_w + 3;

        sum += __ldg(x + x_off0) * __ldg(weight + w_off0);
        sum += __ldg(x + x_off1) * __ldg(weight + w_off1);
        sum += __ldg(x + x_off2) * __ldg(weight + w_off2);
        sum += __ldg(x + x_off3) * __ldg(weight + w_off3);
    }

    // Handle any remaining channels
    for (int ic = ic_unroll; ic < IC; ++ic) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w_pos;
        int w_offset = oc * IC + ic;
        sum += __ldg(x + x_offset) * __ldg(weight + w_offset);
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += __ldg(bias + oc);
    }
    
    output[index] = sum;
}

// CUDA forward function
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

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Launch kernel
    const int threads = 256;
    const int total = B * OC * H * W;
    const int blocks = (total + threads - 1) / threads;
    
    forward_kernel<<<blocks, threads>>>(x_ptr, w_ptr, b_ptr, out_ptr, B, IC, OC, H, W);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA) optimized with __ldg and aligned loads");
}
