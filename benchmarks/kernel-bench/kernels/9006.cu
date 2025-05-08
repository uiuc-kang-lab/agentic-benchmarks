#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_conv1d_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int b,
    int oc,
    int o,
    int in_channels,
    int in_size,
    int kernel_size,
    int stride,
    int dilation,
    bool use_shared_mem) {
    
    __shared__ float shared_weight[32][32];  // Assuming max kernel_size and in_channels <= 32
    float sum = 0.0f;
    int start_pos = o * stride;
    int end_pos = start_pos + (kernel_size - 1) * dilation;
    
    if (use_shared_mem && threadIdx.x < kernel_size && threadIdx.y < in_channels) {
        shared_weight[threadIdx.y][threadIdx.x] = 
            weight[oc * (in_channels * kernel_size) + threadIdx.y * kernel_size + threadIdx.x];
    }
    __syncthreads();

    if (end_pos < in_size) {
        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size + start_pos;
            const float* w_ptr = use_shared_mem ? 
                &shared_weight[ic][0] : 
                weight + oc * (in_channels * kernel_size) + ic * kernel_size;
            
            #pragma unroll
            for (int k = 0; k < kernel_size; k += 4) {
                if (k + 4 <= kernel_size) {
                    sum += x_ptr[k * dilation] * w_ptr[k] +
                          x_ptr[(k+1) * dilation] * w_ptr[k+1] +
                          x_ptr[(k+2) * dilation] * w_ptr[k+2] +
                          x_ptr[(k+3) * dilation] * w_ptr[k+3];
                } else {
                    for (int r = k; r < kernel_size; ++r) {
                        sum += x_ptr[r * dilation] * w_ptr[r];
                    }
                }
            }
        }
    } else {
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size;
            const float* w_ptr = use_shared_mem ? 
                &shared_weight[ic][0] : 
                weight + oc * (in_channels * kernel_size) + ic * kernel_size;
            
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                int pos = start_pos + k * dilation;
                sum += (pos < in_size) * x_ptr[pos] * w_ptr[k];
            }
        }
    }
    return sum;
}

__global__ void conv1d_optimized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * out_channels * out_size) return;

    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    bool use_shared_mem = (kernel_size <= 32 && in_channels <= 32);
    
    float sum = compute_conv1d_optimized(
        x, weight, b, oc, o, in_channels, in_size, 
        kernel_size, stride, dilation, use_shared_mem);

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[b * (out_channels * out_size) + oc * out_size + o] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation) {
    
    TORCH_CHECK(x.device().is_cuda() && weight.device().is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(x.dim() == 3 && weight.dim() == 3, "Inputs must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        auto& bias_t = bias.value();
        TORCH_CHECK(bias_t.device().is_cuda() && bias_t.is_contiguous() && 
                   bias_t.dim() == 1 && bias_t.size(0) == weight.size(0), 
                   "Invalid bias tensor");
    }

    int B = x.size(0), in_channels = x.size(1), in_size = x.size(2);
    int out_channels = weight.size(0), kernel_size = weight.size(2);
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    dim3 threads(256);
    dim3 blocks((B * out_channels * out_size + threads.x - 1) / threads.x);

    conv1d_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, in_channels, in_size, out_channels,
        kernel_size, out_size, stride, dilation);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}