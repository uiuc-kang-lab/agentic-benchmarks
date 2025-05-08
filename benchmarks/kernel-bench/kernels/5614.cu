#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation
) {
    typedef ulonglong2 aligned_type __attribute__((aligned(16)));
    
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;
    const int input_base = b * channels * input_height * input_width 
                         + c * input_height * input_width;

    scalar_t max_val = -__FLT_MAX__;
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for(int kh=0;kh<2;kh++){
            const int ih = h_start + kh*dilation;
            if(ih >=0 && ih<input_height){
                const aligned_type* row = reinterpret_cast<const aligned_type*>(&input[input_base + ih*input_width]);
                #pragma unroll
                for(int kw=0;kw<2;kw++){
                    const int iw = w_start + kw*dilation;
                    if(iw >=0 && iw<input_width){
                        const scalar_t val = __ldg(reinterpret_cast<const scalar_t*>(row) + iw);
                        if(val > max_val) max_val = val;
                    }
                }
            }
        }
    } else {
        for(int kh=0;kh<KERNEL_SIZE;kh++){
            const int ih = h_start + kh*dilation;
            if(ih >=0 && ih<input_height){
                const aligned_type* row = reinterpret_cast<const aligned_type*>(&input[input_base + ih*input_width]);
                for(int kw=0;kw<KERNEL_SIZE;kw++){
                    const int iw = w_start + kw*dilation;
                    if(iw >=0 && iw<input_width){
                        const scalar_t val = __ldg(reinterpret_cast<const scalar_t*>(row) + iw);
                        max_val = max(max_val, val);
                    }
                }
            }
        }
    }

    __syncthreads();
    output[bc*output_height*output_width + oh*output_width + ow] = max_val;
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
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = (input_height + 2*padding - dilation*(kernel_size-1)-1)/stride + 1;
    const auto output_width = (input_width + 2*padding - dilation*(kernel_size-1)-1)/stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options().dtype(input.dtype()).requires_grad(false));

    dim3 threads(16,16);
    dim3 blocks(
        (output_width + threads.x-1)/threads.x,
        (output_height + threads.y-1)/threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool_forward", ([&] {
        if(kernel_size == 2) {
            max_pool2d_optimized_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            max_pool2d_optimized_kernel<scalar_t, -1><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}