#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool2d_shared_kernel(
    const scalar_t* input,
    scalar_t* output,
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
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    const int bid_z = blockIdx.z;

    const int b = bid_z / channels;
    const int c = bid_z % channels;

    const int tile_start_h = bid_y * blockDim.y * stride - padding;
    const int tile_start_w = bid_x * blockDim.x * stride - padding;

    const int tile_end_h = tile_start_h + (blockDim.y * stride) + (kernel_size - 1) * dilation;
    const int tile_end_w = tile_start_w + (blockDim.x * stride) + (kernel_size - 1) * dilation;

    const int tile_size_h = tile_end_h - tile_start_h;
    const int tile_size_w = tile_end_w - tile_start_w;

    for (int i = tid_y; i < tile_size_h; i += blockDim.y) {
        for (int j = tid_x; j < tile_size_w; j += blockDim.x) {
            const int h = tile_start_h + i * dilation;
            const int w = tile_start_w + j * dilation;
            
            if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                sdata[i * tile_size_w + j] = input[b * channels * input_height * input_width
                                                + c * input_height * input_width
                                                + h * input_width
                                                + w];
            } else {
                sdata[i * tile_size_w + j] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }

    __syncthreads();

    const int oh = bid_y * blockDim.y + tid_y;
    const int ow = bid_x * blockDim.x + tid_x;

    if (oh >= output_height || ow >= output_width) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int shmem_h = tid_y * stride + kh;
            const int shmem_w = tid_x * stride + kw;
            if (shmem_h < tile_size_h && shmem_w < tile_size_w) {
                max_val = max(max_val, sdata[shmem_h * tile_size_w + shmem_w]);
            }
        }
    }

    output[b * channels * output_height * output_width
        + c * output_height * output_width
        + oh * output_width
        + ow] = max_val;
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    const int tile_h = block.y * stride + (kernel_size - 1) * dilation;
    const int tile_w = block.x * stride + (kernel_size - 1) * dilation;
    const size_t shared_mem = tile_h * tile_w * sizeof(typename torch::ScalarTypeToCPPType<torch::kFloat>::type);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<grid, block, shared_mem>>>(
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
