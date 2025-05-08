#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each warp computes one output element using warp-level reduction
__global__ void forward_kernel_warp(
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
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp (32 threads) computes one output element
    int warp_id = tid / 32;
    int lane = tid % 32;
    
    const int total_output = B * OC * H * W;
    if (warp_id >= total_output) return;
    
    // Decode 4D output coordinates from warp_id
    int index = warp_id;
    int w_coord = index % W;
    int h_coord = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    // Partition the reduction across the warp's lanes
    for (int ic = lane; ic < IC; ic += 32) {
        int x_index = b * IC * H * W + ic * H * W + h_coord * W + w_coord;
        int weight_index = oc * IC + ic;
        sum += x[x_index] * weight[weight_index];
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Only lane 0 writes the final result
    if (lane == 0) {
        output[index] = (bias != nullptr) ? (sum + bias[oc]) : sum;
    }
}

// Host function wrapping the CUDA kernel
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
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

    // Each output element is computed by one warp (32 threads)
    int total_output = B * OC * H * W;
    int total_threads = total_output * 32;
    const int threads = 256; // block size (multiple of 32)
    int blocks = (total_threads + threads - 1) / threads;

    forward_kernel_warp<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward with warp-level reduction (CUDA)");
}
