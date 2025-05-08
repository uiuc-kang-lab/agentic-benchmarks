#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp-level primitives for reduction
__global__ void avg_pool1d_warp_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    // Determine warp and lane within the warp
    const int warpSize = 32;
    int warps_per_block = blockDim.x / warpSize;
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    // Map each warp to one output element along the length dimension
    int out_idx = blockIdx.x * warps_per_block + warp_id;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (out_idx >= output_length || channel >= in_channels || batch >= batch_size) return;

    float sum = 0.0f;
    // Each thread in the warp sums a subset of the kernel elements
    for (int k = lane; k < kernel_size; k += warpSize) {
        int pos_padded = out_idx * stride + k;
        int pos_input = pos_padded - padding;
        if (pos_input >= 0 && pos_input < input_length) {
            int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
            sum += input[input_idx];
        }
    }

    // Use warp-level reduction to sum partial results
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // The first lane of each warp writes the result
    if (lane == 0) {
        int out_index = batch * in_channels * output_length + channel * output_length + out_idx;
        output[out_index] = sum / kernel_size;
    }
}

// Forward function wrapper
torch::Tensor avg_pool1d_forward_warp(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Launch configuration: each warp computes one output element
    // Select threadsPerBlock as a multiple of warpSize. Here we use 128 (4 warps per block).
    const int threadsPerBlock = 128;
    int warps_per_block = threadsPerBlock / 32;
    int blocks_x = (output_length + warps_per_block - 1) / warps_per_block;

    dim3 threads(threadsPerBlock);
    dim3 blocks(blocks_x, in_channels, batch_size);

    avg_pool1d_warp_kernel<<<blocks, threads>>>(
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
    m.def("forward", &avg_pool1d_forward_warp, "1D Average Pooling forward with warp-level reduction (CUDA)");
}
