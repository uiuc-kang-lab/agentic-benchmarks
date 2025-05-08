#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    const int total = batch * out_channels * out_h * out_w;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int oc = (idx / out_w / out_h) % out_channels;
        output[idx] = __ldg(&bias[oc]);
    }
}

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

    const int total = batch * in_channels * in_h * in_w;
    const int elements_per_thread = 4;
    
    for (int linear_idx = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;
         linear_idx < total;
         linear_idx += blockDim.x * gridDim.x * elements_per_thread) {
        
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            const int index = linear_idx + i * blockDim.x * gridDim.x;
            if (index >= total) break;

            const int iw = index % in_w;
            const int tmp1 = index / in_w;
            const int ih = tmp1 % in_h;
            const int tmp2 = tmp1 / in_h;
            const int c = tmp2 % in_channels;
            const int n = tmp2 / in_channels;

            const float x_val = __ldg(&x[index]);
            const int group = c / in_channels_per_group;
            const int weight_base = c * (out_channels_per_group * kernel_h * kernel_w);
            const int output_batch_offset = n * (groups * out_channels_per_group * out_h * out_w);
            const int group_offset = group * out_channels_per_group;

            #pragma unroll 4
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int out_row = ih * stride_h - pad_h + kh * dilation_h;
                if (out_row < 0 || out_row >= out_h) continue;

                #pragma unroll 4
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int out_col = iw * stride_w - pad_w + kw * dilation_w;
                    if (out_col < 0 || out_col >= out_w) continue;

                    const int kernel_offset = (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);
                    const int weight_start = weight_base + kernel_offset;

                    #pragma unroll 4
                    for (int oc_offset = 0; oc_offset < out_channels_per_group; ++oc_offset) {
                        const float weight_val = __ldg(&weight[weight_start + oc_offset * kernel_h * kernel_w]);
                        const int out_index = output_batch_offset + 
                                            (group_offset + oc_offset) * (out_h * out_w) + 
                                            out_row * out_w + out_col;
                        atomicAdd(&output[out_index], x_val * weight_val);
                    }
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    const int in_channels_per_group = in_channels / groups;
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize output with bias
    const int total_output = batch * out_channels * out_h * out_w;
    const int threads_init = 512;
    const int blocks_init = (total_output + threads_init - 1) / threads_init;
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    // Scatter kernel with grid-stride loops
    const int total_input = batch * in_channels * in_h * in_w;
    const int threads_scatter = 512;
    const int blocks_scatter = min(65535, (total_input + threads_scatter * 4 - 1) / (threads_scatter * 4));
    
    conv_transposed2d_scatter_atomic_kernel<<<blocks_scatter, threads_scatter>>>(
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
    m.def("forward", &forward, "2D Transposed Convolution with Optimized Grid-Stride (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}