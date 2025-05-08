#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ inline void load_input_tile(
    const scalar_t* input,
    scalar_t* shared_data,
    const int tid,
    const int b, const int c,
    const int oh, const int ow,
    const int stride, const int padding,
    const int dilation,
    const int input_height,
    const int input_width,
    const int channels,
    const int TILE_SIZE
) {
    const int ih = oh * stride - padding;
    const int iw = ow * stride - padding;
    
    if (tid < TILE_SIZE * TILE_SIZE) {
        const int local_h = tid / TILE_SIZE;
        const int local_w = tid % TILE_SIZE;
        const int global_h = ih + local_h * dilation;
        const int global_w = iw + local_w * dilation;
        
        if (global_h >= 0 && global_h < input_height && 
            global_w >= 0 && global_w < input_width) {
            const int input_idx = b * (channels * input_height * input_width) +
                                c * (input_height * input_width) +
                                global_h * input_width +
                                global_w;
            shared_data[local_h * TILE_SIZE + local_w] = input[input_idx];
        } else {
            shared_data[local_h * TILE_SIZE + local_w] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
}

template <typename scalar_t>
__device__ inline scalar_t compute_tile_max(
    const scalar_t* shared_data,
    const int kernel_size,
    const int TILE_SIZE
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int shared_idx = kh * TILE_SIZE + kw;
            max_val = max(max_val, shared_data[shared_idx]);
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
    const int TILE_SIZE = 16;  // Adjust based on kernel size and shared memory constraints
    __shared__ scalar_t shared_data[TILE_SIZE * TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= batch_size * channels * output_height * output_width) return;
    
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);
    
    // Load input tile to shared memory
    load_input_tile<scalar_t>(
        input, shared_data, tid,
        b, c, oh, ow, stride, padding, dilation,
        input_height, input_width, channels, TILE_SIZE
    );
    
    __syncthreads();
    
    // Compute max value within the tile
    scalar_t max_val = compute_tile_max<scalar_t>(
        shared_data, kernel_size, TILE_SIZE
    );
    
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

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