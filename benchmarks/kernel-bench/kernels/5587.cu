#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[4];

#define TILE_SIZE_H 8
#define TILE_SIZE_W 8

template <typename scalar_t>
__device__ __forceinline__ bool is_valid_input(
    int h, int w,
    int height, int width
) {
    return h >= 0 && h < height && w >= 0 && w < width;
}

template <typename scalar_t>
__device__ __forceinline__ void load_input_tile(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ shared_input,
    int b, int c,
    int oh_start, int ow_start,
    int input_height, int input_width,
    int channels, int stride, int padding, int dilation
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const int tile_elements = (TILE_SIZE_H + 2) * (TILE_SIZE_W + 2);
    
    #pragma unroll
    for (int i = tid; i < tile_elements; i += block_size) {
        const int tile_h = i / (TILE_SIZE_W + 2);
        const int tile_w = i % (TILE_SIZE_W + 2);
        
        const int ih = oh_start * stride - padding + tile_h;
        const int iw = ow_start * stride - padding + tile_w;
        
        if (is_valid_input<scalar_t>(ih, iw, input_height, input_width)) {
            const int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                ih * input_width + iw;
            shared_input[tile_h * (TILE_SIZE_W + 2) + tile_w] = __ldg(&input[input_idx]);
        } else {
            shared_input[tile_h * (TILE_SIZE_W + 2) + tile_w] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_pool_tile(
    const scalar_t* __restrict__ shared_input,
    int kernel_size,
    int stride,
    int dilation,
    int th, int tw
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int shared_h = th + kh * dilation;
            const int shared_w = tw + kw * dilation;
            max_val = max(max_val, shared_input[shared_h * (TILE_SIZE_W + 2) + shared_w]);
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
    const int output_width
) {
    __shared__ scalar_t shared_input[TILE_SIZE_H + 2][TILE_SIZE_W + 2];
    
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int c = bz % channels;
    const int b = bz / channels;
    
    const int oh_start = by * TILE_SIZE_H;
    const int ow_start = bx * TILE_SIZE_W;
    
    load_input_tile<scalar_t>(
        input, &shared_input[0][0],
        b, c, oh_start, ow_start,
        input_height, input_width,
        channels, stride, padding, dilation
    );
    
    __syncthreads();
    
    if (ty < TILE_SIZE_H && tx < TILE_SIZE_W) {
        const int oh = oh_start + ty;
        const int ow = ow_start + tx;
        
        if (oh < output_height && ow < output_width) {
            const scalar_t max_val = compute_max_pool_tile<scalar_t>(
                &shared_input[0][0],
                kernel_size, stride, dilation,
                ty, tx
            );
            
            const int output_idx = b * (channels * output_height * output_width) +
                                 c * (output_height * output_width) +
                                 oh * output_width + ow;
            output[output_idx] = max_val;
        }
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

    const int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 4);

    dim3 threads(8, 8);
    dim3 blocks(
        (output_width + TILE_SIZE_W - 1) / TILE_SIZE_W,
        (output_height + TILE_SIZE_H - 1) / TILE_SIZE_H,
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
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}