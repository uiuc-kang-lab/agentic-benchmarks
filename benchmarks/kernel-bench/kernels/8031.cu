#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define maximum number of weight elements for constant memory
#define MAX_WEIGHT_SIZE 4096

// Declare constant memory for convolution weights
__constant__ float c_weight[MAX_WEIGHT_SIZE];

// This kernel minimizes warp divergence by replacing conditional branches with branchless arithmetic
// to compute a validity mask. All threads execute a uniform control flow, and invalid computations
// are nullified via multiplication by a 0/1 mask. Safe clamped indices ensure no out-of-bound memory access.

__global__ void warp_divergence_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias, // can be nullptr if no bias
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

    // Grid configuration:
    // blockIdx.x : batch index
    // blockIdx.y : output channel index
    // blockIdx.z : tile index along the output spatial dimension
    int b = blockIdx.x;
    int o = blockIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;  // output spatial index
    if (j >= output_width) return;

    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels / groups;
    int g = o / group_out_channels;       // group index
    int local_o = o % group_out_channels;  // local output channel within group

    float sum = 0.0f;
    // Loop over kernel elements
    for (int k = 0; k < kernel_size; ++k) {
        // Compute the corresponding input spatial index in a branchless manner
        int temp = j + padding - k;
        int div = temp / stride;
        int rem = temp - div * stride;
        // Compute valid flag: 1 if remainder is zero and index within [0, input_width), else 0
        int valid = (int)(rem == 0) * (int)(div >= 0) * (int)(div < input_width);
        // Clamp the division result to a valid range to prevent illegal memory access
        int safe_div = min(max(div, 0), input_width - 1);

        // For each input channel in the current group
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int real_ic = g * group_in_channels + ic;
            int input_index = b * (in_channels * input_width) + real_ic * input_width + safe_div;
            int weight_index = (real_ic * group_out_channels + local_o) * kernel_size + k;
            // Multiply by valid flag; if not valid the product contributes 0
            sum += valid * input[input_index] * c_weight[weight_index];
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[o];
    }

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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_out_channels = weight.size(1);  // weight shape: [in_channels, group_out_channels, kernel_size]
    int out_channels = group_out_channels * groups;
    // Calculate output width for transposed convolution
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    // Copy weight tensor to constant memory
    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Setup grid and block dimensions for coalesced memory access
    int threads = 256; // threads per block for covering output spatial dimension
    dim3 block(threads);
    dim3 grid(
        batch_size,         // each grid.x processes one batch
        out_channels,       // grid.y processes each output channel
        (output_width + threads - 1) / threads  // grid.z tiles over output width
    );

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    warp_divergence_conv1d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    m.def("forward", &forward, "Warp Divergence Optimized Transposed 1D convolution forward (CUDA)");
}
