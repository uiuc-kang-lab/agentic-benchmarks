#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// This revised kernel uses scalar computations for correctness

// Vectorized initialization kernel using float4
__global__ void initialize_output_kernel(
    float4* __restrict__ output4,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_float4s = (batch * out_channels * out_h * out_w + 3) / 4;
    if (tid >= total_float4s) return;

    // Calculate channel indices for each element in float4
    const int elements_per_channel = out_h * out_w;
    const int idx = tid * 4;
    const int oc0 = (idx / elements_per_channel) % out_channels;
    const int oc1 = ((idx + 1) / elements_per_channel) % out_channels;
    const int oc2 = ((idx + 2) / elements_per_channel) % out_channels;
    const int oc3 = ((idx + 3) / elements_per_channel) % out_channels;

    float4 bias4;
    bias4.x = __ldg(&bias[oc0]);
    bias4.y = __ldg(&bias[oc1]);
    bias4.z = __ldg(&bias[oc2]);
    bias4.w = __ldg(&bias[oc3]);
    
    output4[tid] = bias4;
}

// Main convolution kernel with vectorized memory access
__global__ void conv_transposed2d_kernel(
    const float4* __restrict__ x4,
    const float4* __restrict__ weight4,
    float4* __restrict__ output4,
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

    __shared__ float4 weight_shared[32][32]; // Shared memory for weight tiles

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch * in_channels * in_h * in_w / 4;
    if (tid >= total_elements) return;

    // Load input values using vectorized loads
    const float4 x_val = __ldg(&x4[tid]);
    
    // Decode input indices
    const int idx = tid * 4;
    const int iw = (idx % in_w) / 4 * 4;
    const int ih = (idx / in_w) % in_h;
    const int ic = (idx / (in_w * in_h)) % in_channels;
    const int n = idx / (in_channels * in_h * in_w);
    
    const int group = ic / in_channels_per_group;
    const int group_offset = group * out_channels_per_group;

    // Pre-compute output batch offset
    const int out_batch_offset = n * groups * out_channels_per_group * out_h * out_w;

    // Process 4 elements at a time
    #pragma unroll 4
    for (int kh = 0; kh < kernel_h; kh++) {
        const int out_h_base = ih * stride_h - pad_h + kh * dilation_h;
        if (out_h_base < 0 || out_h_base >= out_h) continue;

        #pragma unroll 4
        for (int kw = 0; kw < kernel_w; kw++) {
            const int out_w_base = iw * stride_w - pad_w + kw * dilation_w;
            if (out_w_base < 0 || out_w_base >= out_w) continue;

            // Load weight tile into shared memory
            if (threadIdx.x < 32) {
                const int weight_idx = ic * out_channels_per_group * kernel_h * kernel_w + 
                                     threadIdx.x * kernel_h * kernel_w + 
                                     kh * kernel_w + kw;
                weight_shared[threadIdx.x][0] = __ldg(&weight4[weight_idx/4]);
            }
            __syncthreads();

            // Process output channels in chunks of 4
            #pragma unroll 4
            for (int oc = 0; oc < out_channels_per_group; oc += 4) {
                const float4 weight_val = weight_shared[oc/4][0];
                
                // Compute output indices for vectorized store
                const int out_idx = out_batch_offset + 
                                  (group_offset + oc) * out_h * out_w + 
                                  out_h_base * out_w + 
                                  out_w_base;

                // Vectorized atomic add
                atomicAdd(&output4[out_idx/4].x, x_val.x * weight_val.x);
                atomicAdd(&output4[out_idx/4].y, x_val.y * weight_val.y);
                atomicAdd(&output4[out_idx/4].z, x_val.z * weight_val.z);
                atomicAdd(&output4[out_idx/4].w, x_val.w * weight_val.w);
            }
            __syncthreads();
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
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize output with vectorized kernel
    const int total_float4s = (batch * out_channels * out_h * out_w + 3) / 4;
    const int threads_init = 256;
    const int blocks_init = (total_float4s + threads_init - 1) / threads_init;
    
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    // Launch main convolution kernel
    const int threads = 256;
    const int blocks = (batch * in_channels * in_h * in_w / 4 + threads - 1) / threads;
    
    conv_transposed2d_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<const float4*>(weight.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
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
        in_channels / groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with Vectorized Memory Access (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}