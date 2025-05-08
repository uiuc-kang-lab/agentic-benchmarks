#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for weights (assuming max size of 1024 floats)
__constant__ float d_weights[1024];

// CUDA kernel using stride loops to handle workloads larger than available threads
__global__ void stride_loop_conv1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups,
    int total_elements) {

    // Compute global thread index and stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_val = blockDim.x * gridDim.x;

    // Each thread processes multiple elements using a stride loop
    for (int idx = tid; idx < total_elements; idx += stride_val) {
        // Calculate output indices
        int j = idx % output_width;                       // output spatial position
        int o = (idx / output_width) % out_channels;        // output channel
        int b = idx / (output_width * out_channels);        // batch index

        float sum = 0.0f;
        int group_in_channels = in_channels / groups;
        int group_size_out = out_channels / groups;
        int g = o / group_size_out;  // group index

        // Iterate over kernel elements
        for (int k = 0; k < kernel_size; k++) {
            int i = j + padding - k;
            if (i % stride != 0) continue;
            i /= stride;
            if (i < 0 || i >= input_width) continue;

            // Accumulate over relevant input channels
            for (int ic = 0; ic < group_in_channels; ic++) {
                int real_ic = g * group_in_channels + ic;
                int input_idx = b * in_channels * input_width + real_ic * input_width + i;
                int weight_idx = (real_ic * group_size_out + (o - g * group_size_out)) * kernel_size + k;
                sum += input[input_idx] * d_weights[weight_idx];
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[o];
        }
        
        output[idx] = sum;
    }
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;

    // Compute output width based on transposed convolution formula
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    // Copy weight tensor to constant memory for fast read-only access
    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= 1024, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(d_weights, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int total_elements = batch_size * out_channels * output_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        bias_ptr = bias.value().data_ptr<float>();
    }

    stride_loop_conv1d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups,
        total_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) using stride loops for extended workloads");
}
