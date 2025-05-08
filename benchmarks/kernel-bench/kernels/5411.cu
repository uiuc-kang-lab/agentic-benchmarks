#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ox >= output_width || oy >= output_height) return;

    // Calculate input window start position
    const int ih_start = oy * stride - padding;
    const int iw_start = ox * stride - padding;

    // Determine shared memory tile dimensions
    const int tile_width = blockDim.x * stride + (kernel_size - 1) * dilation;
    const int tile_height = blockDim.y * stride + (kernel_size - 1) * dilation;

    // Load input tile into shared memory
    for (int ty = threadIdx.y; ty < tile_height; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_width; tx += blockDim.x) {
            int ih = ih_start + ty * dilation;
            int iw = iw_start + tx * dilation;
            
            scalar_t val = -std::numeric_limits<scalar_t>::infinity();
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                val = input[((b * channels + c) * input_height + ih) * input_width + iw];
            }
            shared_data[ty * tile_width + tx] = val;
        }
    }
    __syncthreads();

    // Compute max within shared memory tile
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ty = threadIdx.y * stride + kh * dilation;
            int tx = threadIdx.x * stride + kw * dilation;
            if (ty >= 0 && ty < tile_height && tx >= 0 && tx < tile_width) {
                max_val = max(max_val, shared_data[ty * tile_width + tx]);
            }
        }
    }

    // Write final result
    output[((b * channels + c) * output_height + oy) * output_width + ox] = max_val;
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

    // 2D block configuration with shared memory
    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    // Calculate shared memory size per block
    const int tile_width = threads.x * stride + (kernel_size - 1) * dilation;
    const int tile_height = threads.y * stride + (kernel_size - 1) * dilation;
    const size_t shared_mem_size = tile_width * tile_height * sizeof(float);

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
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}
