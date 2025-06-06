#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized device function using vectorized loads where possible
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

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
    int dilation) {
    
    float sum = 0.0f;
    int start_pos = o * stride;
    int end_pos = start_pos + (kernel_size - 1) * dilation;
    
    // Aligned case - use vectorized loads when possible
    if (end_pos < in_size && (kernel_size % 4 == 0) && (dilation == 1) && 
        (((size_t)x & 15) == 0) && (((size_t)weight & 15) == 0)) {
        
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size + start_pos;
            const float* w_ptr = weight + oc * (in_channels * kernel_size) + ic * kernel_size;
            
            #pragma unroll
            for (int k = 0; k < kernel_size; k += 4) {
                float4 x_vec = load_float4(x_ptr + k);
                float4 w_vec = load_float4(w_ptr + k);
                sum += x_vec.x * w_vec.x + x_vec.y * w_vec.y + 
                       x_vec.z * w_vec.z + x_vec.w * w_vec.w;
            }
        }
    }
    // Non-aligned/boundary case - use scalar operations with predication
    else {
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size;
            const float* w_ptr = weight + oc * (in_channels * kernel_size) + ic * kernel_size;
            
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                int pos = start_pos + k * dilation;
                bool valid = pos < in_size;
                sum += valid * x_ptr[pos] * w_ptr[k];
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
    
    // Use shared memory for frequently accessed weight data
    extern __shared__ float shared_weight[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Collaborative loading of weights into shared memory
    if (threadIdx.x < kernel_size * in_channels) {
        shared_weight[threadIdx.x] = weight[oc * (in_channels * kernel_size) + threadIdx.x];
    }
    __syncthreads();

    float sum = compute_conv1d_optimized(x, shared_weight, b, oc, o, 
                                       in_channels, in_size, kernel_size, 
                                       stride, dilation);

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
    
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), "Bias size mismatch");
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int threads = 256;
    int blocks = (B * out_channels * out_size + threads - 1) / threads;
    int shared_mem_size = kernel_size * in_channels * sizeof(float);

    conv1d_optimized_kernel<<<blocks, threads, shared_mem_size>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 1D convolution forward (CUDA)");
}