#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macro to check tensor on CUDA and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA Kernel: Each block computes a tile of the output for one sample and one output channel.
// This implementation leverages shared memory to cache the convolution filter (weight) for the output channel.
// Note: This kernel supports only the case of groups==1 and dilation==1. For other cases, the CPU launches torch::conv2d.

__global__ void conv2d_shared_kernel_optimized(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int stride,
    int padding) {

    // Determine the output channel and batch index from blockIdx.z
    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels;

    // Determine the output x and y positions
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory for the filter weights of current output channel
    extern __shared__ float sh_weight[]; // size: in_channels * kernel_size * kernel_size
    int filter_elems = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Load the kernel filter for output channel 'oc' into shared memory
    // Each thread can load multiple elements if the block is larger than needed
    for (int idx = tid; idx < filter_elems; idx += blockDim.x * blockDim.y) {
        int ic = idx / (kernel_size * kernel_size);
        int rem = idx % (kernel_size * kernel_size);
        int ki = rem / kernel_size;
        int kj = rem % kernel_size;
        // Global index: weight[oc, ic, ki, kj]
        int weight_index = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + ki * kernel_size + kj;
        sh_weight[idx] = weight[weight_index];
    }
    __syncthreads();

    // Check bounds for the output tile
    if (out_row >= out_h || out_col >= out_w) return;

    float sum = 0.0f;
    // For every input channel and kernel element, apply convolution
    for (int ic = 0; ic < in_channels; ic++) {
        for (int ki = 0; ki < kernel_size; ki++) {
            for (int kj = 0; kj < kernel_size; kj++) {
                int in_row = out_row * stride - padding + ki;
                int in_col = out_col * stride - padding + kj;
                if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                    int input_index = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + in_row * in_w + in_col;
                    // Access the filter weight from shared memory
                    int filter_index = ic * (kernel_size * kernel_size) + ki * kernel_size + kj;
                    sum += input[input_index] * sh_weight[filter_index];
                }
            }
        }
    }

    // Use warp-level reduction to sum partial results
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Add bias if provided
    if (bias != nullptr && threadIdx.x % warpSize == 0) { // Only one thread in the warp writes the result
        sum += bias[oc];
    }

    // Write the output: layout [n, oc, out_h, out_w]
    if (threadIdx.x % warpSize == 0) { // Only one thread in the warp writes the result
        int out_index = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + out_row * out_w + out_col;
        output[out_index] = sum;
    }
}


// Host function for the forward pass
// This function checks the input and, if supported parameters are used (groups==1 and dilation==1), launches our custom CUDA kernel
// Otherwise, it falls back to torch::conv2d

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // For simplicity, this kernel supports only groups==1 and dilation==1
    if (groups != 1 || dilation != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }

    // Get input dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Assuming weight layout is [out_channels, in_channels, kernel_size, kernel_size]
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // square kernel assumed

    // Compute output dimensions
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor
    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    // Define block and grid dimensions
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
              batch_size * out_channels);

    // Compute size of shared memory required for the filter
    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);

    // Launch the custom CUDA kernel
    conv2d_shared_kernel_optimized<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_size,
        out_h,
        out_w,
        stride,
        padding
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with shared memory optimization and warp-level reduction");
}