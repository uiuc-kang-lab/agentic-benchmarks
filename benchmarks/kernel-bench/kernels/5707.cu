#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_multi_stream_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_h,
    const int input_w,
    const int output_h,
    const int output_w,
    const int stride,
    const int padding,
    const int dilation,
    const int bc_start,
    const int bc_end
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc_idx = blockIdx.z;
    const int bc_global = bc_start + bc_idx;

    if (ow >= output_w || oh >= output_h || bc_global >= bc_end) return;

    const int b = bc_global / channels;
    const int c = bc_global % channels;
    const int input_offset = (b * channels + c) * input_h * input_w;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int base_h = oh * stride - padding;
    const int base_w = ow * stride - padding;

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        const int ih = base_h + kh * dilation;
        if (0 <= ih && ih < input_h) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int iw = base_w + kw * dilation;
                if (0 <= iw && iw < input_w) {
                    max_val = fmaxf(max_val, __ldg(input + input_offset + ih * input_w + iw));
                }
            }
        }
    }

    output[(bc_global * output_h + oh) * output_w + ow] = max_val;
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
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);

    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int bc_total = batch_size * channels;
    
    auto output = torch::empty({batch_size, channels, output_h, output_w}, input.options());

    const dim3 block(32, 8);
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (auto& stream : streams) cudaStreamCreate(&stream);
    const int chunk_size = (bc_total + num_streams - 1) / num_streams;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        for (int s = 0; s < num_streams; ++s) {
            const int bc_start = s * chunk_size;
            const int bc_end = min(bc_start + chunk_size, bc_total);
            if (bc_start >= bc_end) continue;

            const dim3 grid(
                (output_w + block.x - 1) / block.x,
                (output_h + block.y - 1) / block.y,
                bc_end - bc_start
            );

            switch(kernel_size) {
                case 2:
                    max_pool2d_multi_stream_kernel<scalar_t, 2><<<grid, block, 0, streams[s]>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, channels,
                        input_h, input_w,
                        output_h, output_w,
                        stride, padding, dilation,
                        bc_start, bc_end
                    );
                    break;
                case 3:
                    max_pool2d_multi_stream_kernel<scalar_t, 3><<<grid, block, 0, streams[s]>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, channels,
                        input_h, input_w,
                        output_h, output_w,
                        stride, padding, dilation,
                        bc_start, bc_end
                    );
                    break;
                default:
                    max_pool2d_multi_stream_kernel<scalar_t, -1><<<grid, block, 0, streams[s]>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, channels,
                        input_h, input_w,
                        output_h, output_w,
                        stride, padding, dilation,
                        bc_start, bc_end
                    );
            }
        }
    }));

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Multi-stream coalesced MaxPool2D (CUDA)");
}