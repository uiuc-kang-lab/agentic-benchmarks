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
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    __shared__ scalar_t shared_input[32][32];
    
    const int tile_h_start = (ih_start / 32) * 32;
    const int tile_w_start = (iw_start / 32) * 32;
    
    if (threadIdx.x < kernel_size * kernel_size) {
        int load_h = tile_h_start + threadIdx.x / kernel_size;
        int load_w = tile_w_start + threadIdx.x % kernel_size;
        if (load_h >= 0 && load_h < input_height && load_w >= 0 && load_w < input_width) {
            shared_input[threadIdx.x / kernel_size][threadIdx.x % kernel_size] = 
                input[base_offset + load_h * input_stride_h + load_w];
        }
    }
    __syncthreads();
    
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int h_offset = ih * input_stride_h;
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    if (ih >= tile_h_start && ih < tile_h_start + 32 &&
                        iw >= tile_w_start && iw < tile_w_start + 32) {
                        max_val = max(max_val, shared_input[ih - tile_h_start][iw - tile_w_start]);
                    } else {
                        max_val = max(max_val, __ldg(&input[base_offset + h_offset + iw]));
                    }
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;
    
    if (b >= batch_size) return;

    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int input_stride_h = input_width;

    const int base_offset = b * input_stride_batch + c * input_stride_channel;
    const int ih_start = y * stride - padding;
    const int iw_start = x * stride - padding;

    scalar_t max_val;
    
    switch (kernel_size) {
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
        default:
            max_val = compute_max_generic<scalar_t>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                kernel_size, stride, dilation);
    }

    const int output_idx = b * (channels * output_height * output_width) +
                          c * (output_height * output_width) +
                          y * output_width +
                          x;
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

    const dim3 threads(16, 16);
    const dim3 blocks(
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