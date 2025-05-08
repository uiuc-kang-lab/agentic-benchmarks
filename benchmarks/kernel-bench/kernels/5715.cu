#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_shared_kernel(
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
    const int dilation,
    const int input_tile_h,
    const int input_tile_w
) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_mem);

    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;
    const int block_oh = blockIdx.y * blockDim.y;
    const int block_ow = blockIdx.x * blockDim.x;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    const int input_start_h = block_oh * stride - padding;
    const int input_start_w = block_ow * stride - padding;

    // Cooperative loading of input tile into shared memory
    for (int ih = thread_y; ih < input_tile_h; ih += blockDim.y) {
        for (int iw = thread_x; iw < input_tile_w; iw += blockDim.x) {
            const int src_h = input_start_h + ih * dilation;
            const int src_w = input_start_w + iw * dilation;
            
            if (src_h >= 0 && src_h < input_height && src_w >= 0 && src_w < input_width) {
                shared_input[ih * input_tile_w + iw] = input[(b * channels + c) * input_height * input_width + src_h * input_width + src_w];
            } else {
                shared_input[ih * input_tile_w + iw] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    const int oh = block_oh + thread_y;
    const int ow = block_ow + thread_x;
    if (oh >= output_height || ow >= output_width) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            const int shmem_h = thread_y * stride + kh;
            const int shmem_w = thread_x * stride + kw;
            if (shmem_h < input_tile_h && shmem_w < input_tile_w) {
                max_val = max(max_val, shared_input[shmem_h * input_tile_w + shmem_w]);
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

    const dim3 threads(8, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const int input_tile_h = threads.y * stride + (kernel_size - 1) * dilation;
    const int input_tile_w = threads.x * stride + (kernel_size - 1) * dilation;
    const size_t shared_mem_size = input_tile_h * input_tile_w * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
            dilation,
            input_tile_h,
            input_tile_w
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}