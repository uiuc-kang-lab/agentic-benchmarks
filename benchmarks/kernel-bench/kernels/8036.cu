/*
Optimized Transposed 1D Convolution Kernel
Combines coalesced global memory writes (from Kernel 1) with modular inline index functions (from Kernel 2) and constant memory caching.
*/

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum allowed number of weight elements for constant memory
#define MAX_WEIGHT_SIZE 4096

// Constant memory for convolution weights
__constant__ float d_weight[MAX_WEIGHT_SIZE];

// Inline device functions for computing flattened indices
__device__ inline int get_input_index(int b, int c, int i, int in_channels, int input_width) {
    return b * in_channels * input_width + c * input_width + i;
}

__device__ inline int get_output_index(int b, int o, int j, int out_channels, int output_width) {
    return b * out_channels * output_width + o * output_width + j;
}

// Optimized Kernel: Uses a 3D grid mapping for coalesced global memory writes
// grid.x -> batch index
// grid.y -> output channel index
// grid.z -> tiles covering the output width
// Threads in each block cover consecutive output spatial positions
__global__ void opt_transposed_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,  // Can be nullptr if bias is not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    // Determine output position from grid and thread indices
    int b = blockIdx.x;                     // batch index
    int o = blockIdx.y;                     // output channel index
    int j = blockIdx.z * blockDim.x + threadIdx.x;  // output spatial position
    if (j >= output_width) return;

    // Compute group-related parameters
    int group_size_out = out_channels / groups;       // Number of output channels per group
    int group_in_channels = in_channels / groups;       // Number of input channels per group
    int g = o / group_size_out;                         // Group index
    int o_in_group = o % group_size_out;                // Local output channel index within the group

    float sum = 0.0f;

    // Loop over kernel elements; assume kernel_size is small and unroll if possible
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = j + padding - k;
        // Check if the current output index aligns with stride requirements
        if (in_pos % stride != 0) continue;
        in_pos /= stride;
        if (in_pos < 0 || in_pos >= input_width) continue;

        // Sum over the input channels in the current group
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int input_chan = g * group_in_channels + ic;
            int input_idx = get_input_index(b, input_chan, in_pos, in_channels, input_width);
            // Weight indexing: [in_channels, group_size_out, kernel_size]
            int weight_idx = (input_chan * group_size_out + o_in_group) * kernel_size + k;
            sum += input[input_idx] * d_weight[weight_idx];
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[o];
    }

    // Write the computed value to the output using coalesced memory access
    int output_idx = get_output_index(b, o, j, out_channels, output_width);
    output[output_idx] = sum;
}

// Host wrapper function
torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // Extract input dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);  // weight shape: [in_channels, group_size_out, kernel_size]
    int out_channels = group_size_out * groups;

    // Compute output width for the transposed convolution
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    // Copy weights to constant memory (ensuring the weight number does not exceed MAX_WEIGHT_SIZE)
    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Define block and grid dimensions for coalesced writes in the output
    int threads = 256;  // number of threads per block covering the output width
    dim3 block(threads);
    // Grid dimensions: one block per (batch, output channel, tile along output width)
    dim3 grid(batch_size, out_channels, (output_width + threads - 1) / threads);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Launch the optimized kernel on the current CUDA stream
    opt_transposed_conv1d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed 1D Convolution (CUDA) with coalesced memory accesses and constant memory caching");
}
