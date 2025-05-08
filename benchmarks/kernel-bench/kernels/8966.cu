#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Device function: Decode flat input index to (n, c, ih, iw)
__device__ inline void decode_input_index(const int index, const int in_w, const int in_h, const int in_channels,
                                           int &n, int &c, int &ih, int &iw) {
    iw = index % in_w;
    int tmp = index / in_w;
    ih = tmp % in_h;
    tmp = tmp / in_h;
    c = tmp % in_channels;
    n = tmp / in_channels;
}

// Device function: Compute flat output index given (n, oc, oh, ow)
__device__ inline int compute_output_index(const int n, const int oc, const int oh, const int ow,
                                             const int out_channels, const int out_h, const int out_w) {
    return n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
}

// Device function: Compute flat weight index given (c, oc_offset, kh, kw)
__device__ inline int compute_weight_index(const int c, const int oc_offset,
                                             const int kh, const int kw,
                                             const int kernel_h, const int kernel_w,
                                             const int out_channels_per_group) {
    return c * (out_channels_per_group * kernel_h * kernel_w) +
           oc_offset * (kernel_h * kernel_w) +
           kh * kernel_w + kw;
}

// Device function: Process one input pixel and scatter its contributions
__device__ void process_input_pixel(const float x_val,
                                    const int n, const int c,
                                    const int ih, const int iw,
                                    const int in_channels_per_group,
                                    const int out_channels_per_group,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w,
                                    const int pad_h, const int pad_w,
                                    const int dilation_h, const int dilation_w,
                                    const int groups, const int out_h, const int out_w,
                                    float* output, const float* weight) {
    int group = c / in_channels_per_group;
    int out_channels = groups * out_channels_per_group;
    int output_batch_offset = n * (out_channels * out_h * out_w);
    int group_offset = group * out_channels_per_group;
    
    for (int kh = 0; kh < kernel_h; kh++) {
        int oh = ih * stride_h - pad_h + kh * dilation_h;
        if (oh < 0 || oh >= out_h)
            continue;
        for (int kw = 0; kw < kernel_w; kw++) {
            int ow = iw * stride_w - pad_w + kw * dilation_w;
            if (ow < 0 || ow >= out_w)
                continue;
            for (int oc_offset = 0; oc_offset < out_channels_per_group; oc_offset++) {
                int oc = group_offset + oc_offset;
                int w_idx = compute_weight_index(c, oc_offset, kh, kw, kernel_h, kernel_w, out_channels_per_group);
                float contrib = x_val * weight[w_idx];
                int out_idx = compute_output_index(n, oc, oh, ow, out_channels, out_h, out_w);
                atomicAdd(&output[out_idx], contrib);
            }
        }
    }
}

// Kernel: Each thread processes one input pixel and scatters contributions
__global__ void conv_transposed2d_scatter_atomic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int in_channels, int in_h, int in_w,
    int out_channels_per_group, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups,
    int out_h, int out_w, int in_channels_per_group) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_channels * in_h * in_w;
    if (index >= total)
        return;
    int n, c, ih, iw;
    decode_input_index(index, in_w, in_h, in_channels, n, c, ih, iw);
    float x_val = x[index];
    process_input_pixel(x_val, n, c, ih, iw,
                        in_channels_per_group, out_channels_per_group,
                        kernel_h, kernel_w,
                        stride_h, stride_w,
                        pad_h, pad_w,
                        dilation_h, dilation_w,
                        groups, out_h, out_w,
                        output, weight);
}

// Kernel to initialize the output tensor with bias values
__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int out_channels, int out_h, int out_w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (index >= total)
        return;
    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int oc = tmp % out_channels;
    output[index] = bias[oc];
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
    
    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }
    
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    int in_channels_per_group = in_channels / groups;
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];
    
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;
    
    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());
    
    // Initialize output with bias
    int total_output = batch * out_channels * out_h * out_w;
    int threads_init = 256;
    int blocks_init = (total_output + threads_init - 1) / threads_init;
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch, out_channels, out_h, out_w);
    
    // Launch scatter kernel: each thread handles one input pixel
    int total_input = batch * in_channels * in_h * in_w;
    int threads_scatter = 256;
    int blocks_scatter = (total_input + threads_scatter - 1) / threads_scatter;
    conv_transposed2d_scatter_atomic_kernel<<<blocks_scatter, threads_scatter>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_channels, in_h, in_w,
        out_channels_per_group, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups,
        out_h, out_w, in_channels_per_group);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular 2D Transposed Convolution using Scatter with Atomics (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
