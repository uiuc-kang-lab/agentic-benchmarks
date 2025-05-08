#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

template<typename scalar_t, int KERNEL_SIZE>
__device__ scalar_t compute_pool_max(
    const scalar_t* input,
    int b, int c,
    int input_h, int input_w,
    int oh, int ow,
    int stride, int padding, int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih < 0 || ih >= input_h) continue;
        
        #pragma unroll
        for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
            int iw = ow * stride - padding + kw * dilation;
            if (iw >= 0 && iw < input_w) {
                scalar_t val = __ldg(&input[
                    b * input_h * input_w + c * input_h * input_w +
                    ih * input_w + iw
                ]);
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    // Warp-level max reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        scalar_t shuffled = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        max_val = fmaxf(max_val, shuffled);
    }
    
    return max_val;
}

template<typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w,
    int stride, int padding, int dilation
) {
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int b = blockIdx.z;
    
    if (b >= batch_size || c >= channels) return;
    
    for (int idx = hw; idx < output_h * output_w; idx += blockDim.x * gridDim.x) {
        int oh = idx / output_w;
        int ow = idx % output_w;
        
        scalar_t final_max = compute_pool_max<scalar_t, KERNEL_SIZE>(
            input, b, c, input_h, input_w,
            oh, ow, stride, padding, dilation
        );
        
        if (threadIdx.x % WARP_SIZE == 0) {
            output[b * output_h * output_w + c * output_h * output_w + idx] = final_max;
        }
    }
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_h, output_w}, input.options());
    
    dim3 blocks((output_h * output_w + 255) / 256, channels, batch_size);
    dim3 threads(256);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", [&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_warp_kernel<scalar_t, 2><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_h, input_w,
                    output_h, output_w, stride, padding, dilation
                );
                break;
            case 3:
                max_pool2d_warp_kernel<scalar_t, 3><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_h, input_w,
                    output_h, output_w, stride, padding, dilation
                );
                break;
            default:
                AT_ERROR("Unsupported kernel size");
        }
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with warp reduction (CUDA)");
}