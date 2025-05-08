#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void shared_warp_maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;

    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;
    
    const int input_offset = b * channels * input_height * input_width 
                           + c * input_height * input_width;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        thread_max = max(thread_max, 
                            __ldg(&input[input_offset + ih * input_width + iw]));
                    }
                }
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        thread_max = max(thread_max, 
                            __ldg(&input[input_offset + ih * input_width + iw]));
                    }
                }
            }
        }
    } else {
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        thread_max = max(thread_max, 
                            __ldg(&input[input_offset + ih * input_width + iw]));
                    }
                }
            }
        }
    }

    // Store in shared memory
    shared_data[tid] = thread_max;
    __syncthreads();

    // Warp-level reduction
    if (lane_id < 16) shared_data[tid] = max(shared_data[tid], shared_data[tid + 16]);
    if (lane_id < 8)  shared_data[tid] = max(shared_data[tid], shared_data[tid + 8]);
    if (lane_id < 4)  shared_data[tid] = max(shared_data[tid], shared_data[tid + 4]);
    if (lane_id < 2)  shared_data[tid] = max(shared_data[tid], shared_data[tid + 2]);
    if (lane_id < 1)  shared_data[tid] = max(shared_data[tid], shared_data[tid + 1]);

    if (tid == 0) {
        const int out_idx = bc * output_height * output_width + oh * output_width + ow;
        output[out_idx] = shared_data[0];
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

    const dim3 threads(16, 16);  // 256 threads per block
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const int shared_memory_size = threads.x * threads.y * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            shared_warp_maxpool2d_kernel<scalar_t, 2><<<blocks, threads, shared_memory_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            shared_warp_maxpool2d_kernel<scalar_t, 3><<<blocks, threads, shared_memory_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            shared_warp_maxpool2d_kernel<scalar_t, -1><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}