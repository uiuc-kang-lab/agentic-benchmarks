#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t, int KERNEL_SIZE>
__device__ __forceinline__ scalar_t compute_max_aligned(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int ih_start,
    const int iw_start,
    const int input_height,
    const int input_width,
    const int input_stride_h,
    const int stride,
    const int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int h_offset = ih * input_stride_h;
            
            // Ensure aligned access for float4 loads when possible
            const int aligned_iw = (iw_start + 3) & ~3;
            const int end_aligned_iw = iw_start + KERNEL_SIZE * dilation;
            
            // Handle pre-aligned elements
            #pragma unroll
            for (int iw = iw_start; iw < min(aligned_iw, end_aligned_iw); iw++) {
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[base_offset + h_offset + iw]));
                }
            }
            
            // Vector loads for aligned portion
            if (sizeof(scalar_t) == 4 && aligned_iw + 4 <= end_aligned_iw) {
                const float4* aligned_input = reinterpret_cast<const float4*>(&input[base_offset + h_offset + aligned_iw]);
                float4 vec_val = __ldg(aligned_input);
                max_val = max(max_val, max(max(vec_val.x, vec_val.y), max(vec_val.z, vec_val.w)));
            }
            
            // Handle remaining elements
            #pragma unroll
            for (int iw = aligned_iw + 4; iw < end_aligned_iw; iw++) {
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
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride_x = blockDim.x * gridDim.x;
    
    const int total_elements = batch_size * channels * output_height * output_width;
    
    for (int index = tid; index < total_elements; index += stride_x) {
        const int ow = index % output_width;
        const int oh = (index / output_width) % output_height;
        const int c = (index / (output_width * output_height)) % channels;
        const int b = index / (output_width * output_height * channels);

        const int input_stride_batch = channels * input_height * input_width;
        const int input_stride_channel = input_height * input_width;
        const int input_stride_h = input_width;

        const int base_offset = b * input_stride_batch + c * input_stride_channel;
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;

        scalar_t max_val;
        
        switch(kernel_size) {
            case 2:
                max_val = compute_max_aligned<scalar_t, 2>(
                    input, base_offset, ih_start, iw_start,
                    input_height, input_width, input_stride_h,
                    stride, dilation);
                break;
            case 3:
                max_val = compute_max_aligned<scalar_t, 3>(
                    input, base_offset, ih_start, iw_start,
                    input_height, input_width, input_stride_h,
                    stride, dilation);
                break;
            default:
                max_val = compute_max_aligned<scalar_t, 4>(
                    input, base_offset, ih_start, iw_start,
                    input_height, input_width, input_stride_h,
                    stride, dilation);
        }
        
        output[index] = max_val;
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
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = min(65535, (batch_size * channels * output_height * output_width + threads - 1) / threads);

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