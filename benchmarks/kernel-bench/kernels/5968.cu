#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses warp-level primitives for reduction in average pooling
__global__ void warp_avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    const int warpSize = 32;
    int warpId = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    // Each warp processes one output element
    int warpsPerBlock = blockDim.x / warpSize;
    int warp_global = blockIdx.x * warpsPerBlock + warpId;

    if (warp_global >= output_length) return;

    // Get channel and batch indices from grid dimensions
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    // Calculate starting index for the pooling window
    int start = warp_global * stride - padding;
    float sum = 0.0f;

    // Each thread in the warp loads one or more pooling elements
    for (int k = lane; k < kernel_size; k += warpSize) {
        int pos = start + k;
        if (pos >= 0 && pos < input_length) {
            int idx = batch * in_channels * input_length + channel * input_length + pos;
            sum += input[idx];
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // The first thread in the warp writes the result
    if (lane == 0) {
        int out_idx = batch * in_channels * output_length + channel * output_length + warp_global;
        output[out_idx] = sum / kernel_size;
    }
}

// Host function to launch the warp-level average pooling kernel
torch::Tensor warp_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Configure launch parameters: assign each warp to one output element
    int warpsPerBlock = 4; // e.g., 4 warps per block (128 threads per block)
    int threadsPerBlock = warpsPerBlock * 32;
    int gridDimX = (output_length + warpsPerBlock - 1) / warpsPerBlock;

    dim3 blocks(gridDimX, in_channels, batch_size);
    dim3 threads(threadsPerBlock);

    warp_avg_pool1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with warp-level reduction");
}
