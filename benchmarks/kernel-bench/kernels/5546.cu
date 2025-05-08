#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t>
__device__ __forceinline__ float4 load_vector(const scalar_t* __restrict__ ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t max4(const float4& v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_aligned(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int ih_start,
    const int iw_start,
    const int input_height,
    const int input_width,
    const int input_stride_h,
    const int kernel_size,
    const int stride,
    const int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int h_offset = ih * input_stride_h;
            
            // Check if we can do aligned vector loads
            const int aligned_start = (iw_start + 3) & ~3;
            const int aligned_end = (iw_start + kernel_size) & ~3;
            
            // Handle initial unaligned elements
            #pragma unroll
            for (int iw = iw_start; iw < min(aligned_start, iw_start + kernel_size); iw++) {
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[base_offset + h_offset + iw]));
                }
            }
            
            // Vector loads for aligned portion
            for (int iw = aligned_start; iw < aligned_end; iw += 4) {
                if (iw + 3 < input_width) {
                    float4 v = load_vector(&input[base_offset + h_offset + iw]);
                    max_val = max(max_val, max4(v));
                }
            }
            
            // Handle final unaligned elements
            #pragma unroll
            for (int iw = aligned_end; iw < iw_start + kernel_size; iw++) {
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[base_offset + h_offset + iw]));
                }
            }
        }
    }
    return max_val;
}

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int ow = bx * blockDim.x + tx;
    const int oh = by * blockDim.y + ty;
    
    if (ow >= output_width || oh >= output_height) return;
    
    const int c = bz % channels;
    const int b = bz / channels;
    
    if (b >= batch_size) return;

    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int input_stride_h = (input_width + 3) & ~3;

    const int base_offset = b * input_stride_batch + c * input_stride_channel;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    scalar_t max_val = compute_max_aligned<scalar_t>(
        input, base_offset, ih_start, iw_start,
        input_height, input_width, input_stride_h,
        kernel_size, stride, dilation
    );

    const int output_idx = ((b * channels + c) * output_height + oh) * output_width + ow;
    output[output_idx] = max_val;
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

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}