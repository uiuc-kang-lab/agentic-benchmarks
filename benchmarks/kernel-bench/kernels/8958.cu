#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Kernel to initialize the output tensor with bias values (or zero if bias is not provided).
__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (index >= total) return;

    // Decode flat index into (n, oc, oh, ow)
    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int oc = tmp % out_channels;

    // Each output element is initialized to the corresponding bias value.
    output[index] = bias[oc];
}

// Scatter-based 2D transposed convolution kernel using atomicAdd to handle overlapping writes.
// Each thread processes one input pixel and scatters its contributions to the output.
__global__ void conv_transposed2d_scatter_atomic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels_per_group,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_h,
    const int out_w,
    const int in_channels_per_group) {

    extern __shared__ float shared_weight[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_channels * in_h * in_w;
    if (index >= total) return;

    // Decode input index
    int iw = index % in_w;
    int tmp = index / in_w;
    int ih = tmp % in_h;
    tmp = tmp / in_h;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    float x_val = x[index];
    int group = c / in_channels_per_group;

    // Load weights into shared memory
    int weight_offset = c * (out_channels_per_group * kernel_h * kernel_w);
    for (int i = threadIdx.x; i < out_channels_per_group * kernel_h * kernel_w; i += blockDim.x) {
        shared_weight[i] = weight[weight_offset + i];
    }
    __syncthreads();

    // Iterate over the kernel spatial dimensions
    for (int kh = 0; kh < kernel_h; kh++) {
        int out_row = ih * stride_h - pad_h + kh * dilation_h;
        if (out_row < 0 || out_row >= out_h) continue;
        for (int kw = 0; kw < kernel_w; kw++) {
            int out_col = iw * stride_w - pad_w + kw * dilation_w;
            if (out_col < 0 || out_col >= out_w) continue;

            int kernel_offset = kh * kernel_w + kw;

            for (int oc_offset = 0; oc_offset < out_channels_per_group; oc_offset++) {
                int oc = group * out_channels_per_group + oc_offset;

                float contrib = x_val * shared_weight[oc_offset * (kernel_h * kernel_w) + kernel_offset];

                int out_index = n * (groups * out_channels_per_group * out_h * out_w) +
                                oc * (out_h * out_w) +
                                out_row * out_w + out_col;

                atomicAdd(&output[out_index], contrib);
            }
        }
    }
}

// Forward function for 2D transposed convolution using the scatter atomic kernel.
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {

    // Ensure inputs are contiguous
    x = x.contiguous();
    weight = weight.contiguous();

    // If bias is not provided, create a tensor of zeros
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }

    // Input dimensions
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    // Weight dimensions
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    const int in_channels_per_group = in_channels / groups;

    // Convolution parameters
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    // Compute output dimensions
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    // Create output tensor
    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize the output with bias values using the initialize kernel
    int total_output = batch * out_channels * out_h * out_w;
    const int threads_init = 256;
    const int blocks_init = (total_output + threads_init - 1) / threads_init;
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    // Launch scatter kernel: each thread handles one input pixel
    int total_input = batch * in_channels * in_h * in_w;
    const int threads_scatter = 256;
    const int blocks_scatter = (total_input + threads_scatter - 1) / threads_scatter;

    size_t shared_memory_size = out_channels_per_group * kernel_h * kernel_w * sizeof(float);

    conv_transposed2d_scatter_atomic_kernel<<<blocks_scatter, threads_scatter, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_h,
        out_w,
        in_channels_per_group);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution using Scatter with Atomics (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
