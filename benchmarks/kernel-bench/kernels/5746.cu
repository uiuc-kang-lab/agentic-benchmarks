#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename scalar_t>
__device__ void process_2x2_warp(
    const scalar_t* input_channel,
    scalar_t& max_val,
    int base_ih,
    int base_iw,
    int dilation,
    int input_height,
    int input_width,
    int lane_id
) {
    const int kh = lane_id / 2;
    const int kw = lane_id % 2;
    
    if (lane_id < 4) {
        const int ih = base_ih + kh * dilation;
        const int iw = base_iw + kw * dilation;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            max_val = input_channel[ih * input_width + iw];
        }
    }
    
    max_val = warp_reduce_max(max_val);
}

template <typename scalar_t>
__device__ void process_3x3_warp(
    const scalar_t* input_channel,
    scalar_t& max_val,
    int base_ih,
    int base_iw,
    int dilation,
    int input_height,
    int input_width,
    int lane_id
) {
    const int kh = lane_id / 3;
    const int kw = lane_id % 3;
    
    if (lane_id < 9) {
        const int ih = base_ih + kh * dilation;
        const int iw = base_iw + kw * dilation;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            max_val = input_channel[ih * input_width + iw];
        }
    }
    
    max_val = warp_reduce_max(max_val);
}

template <typename scalar_t>
__device__ void process_generic(
    const scalar_t* input_channel,
    scalar_t& max_val,
    int base_ih,
    int base_iw,
    int kernel_size,
    int dilation,
    int input_height,
    int input_width
) {
    #pragma unroll 4
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = base_ih + kh * dilation;
        const bool valid_h = ih >= 0 && ih < input_height;
        
        #pragma unroll 4
        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw = base_iw + kw * dilation;
            if (valid_h && iw >= 0 && iw < input_width) {
                max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    }
}

template <typename scalar_t>
__global__ void max_pool2d_warp_kernel(
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
    const int lane_id = threadIdx.x % 32;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;

    if (kernel_size == 2) {
        process_2x2_warp(input_channel, max_val, base_ih, base_iw,
                        dilation, input_height, input_width, lane_id);
    }
    else if (kernel_size == 3) {
        process_3x3_warp(input_channel, max_val, base_ih, base_iw,
                        dilation, input_height, input_width, lane_id);
    }
    else {
        process_generic(input_channel, max_val, base_ih, base_iw,
                       kernel_size, dilation, input_height, input_width);
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

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_warp_kernel<scalar_t><<<blocks, threads>>>(
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