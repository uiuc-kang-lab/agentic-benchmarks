#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized CUDA kernel using shared memory and warp-level primitives
__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    extern __shared__ float shared_input[];
    
    int n = blockIdx.x;
    int oc = blockIdx.y;
    int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;
    
    const int in_y_start = out_y * stride - padding;
    const int in_x_start = out_x * stride - padding;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = in_y_start + ky * dilation;
                int in_x = in_x_start + kx * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    shared_input[threadIdx.y * blockDim.x + threadIdx.x] = 
                        input[n * in_channels * in_height * in_width +
                              ic * in_height * in_width +
                              in_y * in_width + in_x];
                } else {
                    shared_input[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
                }
            }
            __syncthreads();
            
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                sum += shared_input[threadIdx.y * blockDim.x + threadIdx.x] *
                       weight[oc * in_channels * kernel_size * kernel_size +
                             ic * kernel_size * kernel_size +
                             ky * kernel_size + kx];
            }
            __syncthreads();
        }
    }

    if (bias) {
        sum += bias[oc];
    }
    
    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        output[n * out_channels * out_height * out_width +
               oc * out_height * out_width +
               out_y * out_width + out_x] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());
    
    dim3 threads(out_width, 1);
    dim3 blocks(batch, out_channels, (out_height + threads.y - 1) / threads.y);
    
    const size_t shared_memory_size = threads.x * threads.y * sizeof(float);
    
    conv2d_optimized_kernel<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);
        
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA convolution with shared memory and warp-level primitives");
}