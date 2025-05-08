#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#define MAX_POOL_SIZE 81
#define SMALL_KERNEL_THRESHOLD 3

struct PoolOffset {
    int dr;
    int dc;
};

__constant__ PoolOffset d_pool_offsets[MAX_POOL_SIZE];
__constant__ int d_pool_offsets_count;

template <typename scalar_t, int KERNEL_SIZE>
__device__ __forceinline__ void process_small_kernel(
    const scalar_t* __restrict__ input_channel,
    const int oh, const int ow,
    const int input_height, const int input_width,
    const int stride, const int padding, const int dilation,
    scalar_t& max_val
) {
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        #pragma unroll
        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
            const int ih = oh * stride - padding + kh * dilation;
            const int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    }
}

template <typename scalar_t>
__global__ void max_pool2d_hybrid_kernel(
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
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

    if (kernel_size == 2) {
        process_small_kernel<scalar_t, 2>(input_channel, oh, ow, input_height, input_width, 
                                        stride, padding, dilation, max_val);
    }
    else if (kernel_size == 3) {
        process_small_kernel<scalar_t, 3>(input_channel, oh, ow, input_height, input_width, 
                                        stride, padding, dilation, max_val);
    }
    else {
        const int base_ih = oh * stride - padding;
        const int base_iw = ow * stride - padding;
        
        for (int i = 0; i < d_pool_offsets_count; i++) {
            int ih = base_ih + d_pool_offsets[i].dr;
            int iw = base_iw + d_pool_offsets[i].dc;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    }

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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

    if (kernel_size > SMALL_KERNEL_THRESHOLD) {
        if (kernel_size * kernel_size > MAX_POOL_SIZE) {
            throw std::runtime_error("Kernel size too large for constant memory pool offsets");
        }
        
        PoolOffset h_pool_offsets[MAX_POOL_SIZE];
        int count = kernel_size * kernel_size;
        for (int r = 0; r < kernel_size; r++) {
            for (int c = 0; c < kernel_size; c++) {
                h_pool_offsets[r * kernel_size + c].dr = r * dilation;
                h_pool_offsets[r * kernel_size + c].dc = c * dilation;
            }
        }
        cudaMemcpyToSymbol(d_pool_offsets, h_pool_offsets, count * sizeof(PoolOffset));
        cudaMemcpyToSymbol(d_pool_offsets_count, &count, sizeof(int));
    }

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_hybrid_kernel<scalar_t><<<blocks, threads>>>(
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