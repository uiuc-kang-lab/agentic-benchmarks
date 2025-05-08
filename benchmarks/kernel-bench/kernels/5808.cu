#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Use shared memory for block-wise reduction and warp-level primitives for final stages

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
    extern __shared__ scalar_t shared_max[];
    
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int batch_offset = blockIdx.z * channels * input_height * input_width;
    const int channel_offset = (blockIdx.z % channels) * input_height * input_width;
    const int w_idx = blockIdx.x * blockDim.x + tid;
    const int h_idx = blockIdx.y;

    if (w_idx >= output_width || h_idx >= output_height) return;

    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = h_idx * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = w_idx * stride - padding + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    int input_idx = batch_offset + channel_offset + ih * input_width + iw;
                    local_max = max(local_max, input[input_idx]);
                }
            }
        }
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Perform warp reduction within each block
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t val = __shfl_down_sync(0xffffffff, shared_max[tid], offset);
        if (lane + offset < blockDim.x) {
            shared_max[tid] = max(shared_max[tid], val);
        }
        __syncthreads();
    }

    // The first thread of each warp stores the maximal value
    if (lane == 0) {
        output[blockIdx.z * (channels * output_height * output_width) + (blockIdx.z % channels) * (output_height * output_width) + h_idx * output_width + w_idx] = shared_max[tid];
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

    const dim3 threads(32, 1, 1);
    const dim3 blocks((output_width + threads.x - 1) / threads.x, output_height, batch_size * channels);
    const int shared_mem_size = threads.x * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}