#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum allowed number of weight elements for constant memory
#define MAX_WEIGHT_SIZE 4096

// Constant memory for the convolution weights
__constant__ float c_weight[MAX_WEIGHT_SIZE];

// Kernel: Designed to ensure memory coalescing by aligning global memory accesses.
// Each block is mapped to a specific batch and output channel, and threads within a block
// cover consecutive output spatial positions, thus writing to contiguous global memory locations.

__global__ void coalesced_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,  // Can be nullptr if not provided
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

    // Grid dimensions:
    // blockIdx.x -> batch index
    // blockIdx.y -> output channel index
    // blockIdx.z -> tile index along the output width dimension
    
    int b = blockIdx.x;             // batch index
    int o = blockIdx.y;             // output channel index
    int j = blockIdx.z * blockDim.x + threadIdx.x;  // output spatial position

    if (j >= output_width) return;

    // Determine the group this output channel belongs to
    int group_size_out = out_channels / groups;         // number of output channels per group
    int group_in_channels = in_channels / groups;         // number of input channels per group
    int g = o / group_size_out;                           // group index
    int o_in_group = o % group_size_out;                  // local output channel index within the group

    float sum = 0.0f;
    // Loop over kernel width
    for (int k = 0; k < kernel_size; ++k) {
        int i_val = j + padding - k;
        // Only valid if the computed index aligns with the stride
        if (i_val % stride != 0) continue;
        int i_idx = i_val / stride;
        if (i_idx < 0 || i_idx >= input_width) continue;
        
        // Sum over all input channels in the group
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int input_channel = g * group_in_channels + ic;
            // Compute input index (batch, channel, spatial)
            int input_index = b * (in_channels * input_width) + input_channel * input_width + i_idx;
            
            // Weight indexing: weight shape is assumed to be (in_channels, group_size_out, kernel_size)
            int weight_index = (input_channel * group_size_out + o_in_group) * kernel_size + k;
            
            sum += input[input_index] * c_weight[weight_index];
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[o];
    }

    // Write the result to output (output layout: [batch, out_channels, output_width])
    int output_index = b * (out_channels * output_width) + o * output_width + j;
    output[output_index] = sum;
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

    // Dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1); // weight shape: [in_channels, group_size_out, kernel_size]
    int out_channels = group_size_out * groups;

    // Calculate output width for transposed convolution
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    // Copy weights to constant memory (ensuring the weight number does not exceed MAX_WEIGHT_SIZE)
    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Define block and grid dimensions to ensure coalesced accesses in the output
    // Blocks: one block per (batch, output_channel, tile along output width)
    int threads = 256;  // number of threads along the output width dimension
    dim3 block(threads);
    dim3 grid(
        batch_size,         // each block in x handles one batch
        out_channels,       // each block in y handles one output channel
        (output_width + threads - 1) / threads  // tile over output width
    );

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Launch the kernel on the current CUDA stream
    coalesced_conv1d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    m.def("forward", &forward, "Coalesced Memory Transposed 1D convolution forward (CUDA)\n"
    "using constant memory and 3D grid for memory coalescing");
}
