#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t, int KERNEL_SIZE>
__device__ __forceinline__ scalar_t compute_max_small_kernel(
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
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    max_val = max(max_val, __ldg(&input[base_offset + h_offset + iw]));
                }
            }
        }
    }
    return max_val;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_generic(
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
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    const int tid = threadIdx.x;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    const int chunk_size = blockDim.x;
    const int num_elements = kernel_size * kernel_size;
    
    for (int i = tid; i < num_elements; i += chunk_size) {
        const int kh = i / kernel_size;
        const int kw = i % kernel_size;
        const int ih = ih_start + kh * dilation;
        const int iw = iw_start + kw * dilation;
        
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            shared_data[i] = __ldg(&input[base_offset + ih * input_stride_h + iw]);
        } else {
            shared_data[i] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
    __syncthreads();
    
    for (int i = 0; i < num_elements; i++) {
        max_val = max(max_val, shared_data[i]);
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
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels * output_height * output_width) return;

    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int input_stride_h = input_width;

    const int ow = index % output_width;
    const int oh = (index / output_width) % output_height;
    const int c = (index / (output_width * output_height)) % channels;
    const int b = index / (output_width * output_height * channels);

    const int base_offset = b * input_stride_batch + c * input_stride_channel;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    scalar_t max_val;
    
    switch(kernel_size) {
        case 2:
            max_val = compute_max_small_kernel<scalar_t, 2>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                stride, dilation);
            break;
        case 3:
            max_val = compute_max_small_kernel<scalar_t, 3>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                stride, dilation);
            break;
        case 4:
            max_val = compute_max_small_kernel<scalar_t, 4>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                stride, dilation);
            break;
        default:
            max_val = compute_max_generic<scalar_t>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                kernel_size, stride, dilation);
    }

    output[index] = max_val;
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
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    const int shared_mem_size = kernel_size * kernel_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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