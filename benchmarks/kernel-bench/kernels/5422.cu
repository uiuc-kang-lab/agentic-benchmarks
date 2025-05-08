#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__inline__ __device__
scalar_t warpReduceMax(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template <typename scalar_t>
__global__ void max_pool2d_kernel_warp(
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
    __shared__ scalar_t shared_data[32][33];  // +1 to avoid bank conflicts
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ox >= output_width || oy >= output_height) return;

    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();

    // Calculate window boundaries
    const int start_h = oy * stride - padding;
    const int start_w = ox * stride - padding;

    // Each thread processes its portion of the pooling window
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = start_h + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = start_w + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                    thread_max = max(thread_max, input[input_idx]);
                }
            }
        }
    }

    // Warp-level reduction
    thread_max = warpReduceMax(thread_max);

    // First thread in warp writes result
    if (lane_id == 0) {
        shared_data[warp_id][0] = thread_max;
    }
    __syncthreads();

    // Final reduction across warps (if needed)
    if (tid < (blockDim.x * blockDim.y + 31) / 32) {
        thread_max = shared_data[tid][0];
    }
    
    if (tid == 0) {
        const int output_idx = ((b * channels + c) * output_height + oy) * output_width + ox;
        output[output_idx] = thread_max;
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

    dim3 threads(32, 8);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_warp<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with warp reduction (CUDA)");
}