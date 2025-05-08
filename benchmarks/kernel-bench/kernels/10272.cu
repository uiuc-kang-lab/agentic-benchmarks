#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a 2D grid so that threads in the x-dimension process contiguous
// output pixels along the width (the fastest changing memory dimension). This ensures
// memory coalescing for both the input and output accesses.

__global__ void forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    // Each block in grid.y corresponds to a unique (b, oc, h) combination.
    int outer_idx = blockIdx.y + blockIdx.z * gridDim.y;
    int h = outer_idx % H;
    int oc = (outer_idx / H) % OC;
    int b = outer_idx / (H * OC);

    // Threads in block.x cover the width dimension, which is contiguous in memory.
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= W) return;

    float sum = 0.0f;
    // Compute the 1x1 convolution (pointwise) by doing a dot product over the input channels
    for (int ic = 0; ic < IC; ++ic) {
        // For input tensor x with layout [B, IC, H, W], the spatial row (w) is contiguous.
        int x_idx = b * IC * H * W + ic * H * W + h * W + w;
        int weight_idx = oc * IC + ic;
        sum += x[x_idx] * weight[weight_idx];
    }

    // Apply bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write to output tensor which has layout [B, OC, H, W]
    int out_idx = b * OC * H * W + oc * H * W + h * W + w;
    output[out_idx] = sum;
}


// CUDA forward function
// Launch configuration is chosen so that each thread block covers a row (across W) of a given pixel location
// for a fixed (b, oc, h) so that global memory accesses are coalesced.

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

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Set up a 2D grid configuration:
    //  - The x-dimension covers the width dimension (W) with threads per block chosen to be 32 (warp size).
    //  - The y-dimension covers each unique (b, oc, h) combination, ensuring that consecutive threads in x
    //    access consecutive memory locations for both x and output.
    const int threads_x = 32;
    dim3 blockDim(threads_x);
    dim3 gridDim((W + threads_x - 1) / threads_x, B * OC * H);

    forward_kernel<<<gridDim, blockDim>>>(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Coalesced Pointwise 2D convolution forward (CUDA)");
}
