#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define maximum constant memory space for weights (in floats)
#define MAX_CONST_WEIGHT_SIZE 4096

// Store frequently accessed weight data in constant memory
__constant__ float c_weight[MAX_CONST_WEIGHT_SIZE];

// CUDA kernel with loop unrolling for inner loops
__global__ void const_mem_unroll_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
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

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_width;
    if (index >= total) return;

    // Decode output indices: b = batch, o = output channel, j = spatial position
    int j = index % output_width;
    int o = (index / output_width) % out_channels;
    int b = index / (output_width * out_channels);

    int group_in_channels = in_channels / groups;
    int group_size_out = out_channels / groups;
    int g = o / group_size_out;  // group index

    float sum = 0.0f;

    // Loop over the kernel elements with pragma unroll for small kernel sizes
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int i = j + padding - k;
        // Only process positions that align with the stride
        if (i % stride != 0) continue;
        i /= stride;
        if (i < 0 || i >= input_width) continue;

        // Loop over the input channels in the current group
        #pragma unroll
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int real_ic = g * group_in_channels + ic;
            int input_index = b * in_channels * input_width + real_ic * input_width + i;
            int weight_index = (real_ic * group_size_out + (o - g * group_size_out)) * kernel_size + k;
            sum += input[input_index] * c_weight[weight_index];
        }
    }

    // Add bias if provided
    if (bias != nullptr) sum += bias[o];

    int output_index = b * out_channels * output_width + o * output_width + j;
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
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;
    // Calculate output width from transposed conv formula
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_CONST_WEIGHT_SIZE, "Weight size exceeds constant memory limits");
    
    // Copy weights into constant memory for fast, cached access
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int total_threads = batch_size * out_channels * output_width;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    const_mem_unroll_conv1d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with constant memory and unrolled loops");
}
