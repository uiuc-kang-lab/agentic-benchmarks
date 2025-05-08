#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void maxpool2d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int channels,
    const int batch_size) {

    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);

    const int tile_width = blockDim.x + kernel_size - 1;
    const int tile_height = blockDim.y + kernel_size - 1;
    
    const int x = blockIdx.x * blockDim.x;
    const int y = blockIdx.y * blockDim.y;
    const int z = blockIdx.z;

    // Load input tile into shared memory
    for (int i = threadIdx.y; i < tile_height; i += blockDim.y) {
        for (int j = threadIdx.x; j < tile_width; j += blockDim.x) {
            const int ih = y * stride + i * dilation - padding;
            const int iw = x * stride + j * dilation - padding;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                shared_data[i * tile_width + j] = input[z * (input_height * input_width * channels) + 
                                                         blockIdx.z * (input_height * input_width) + 
                                                         ih * input_width + iw];
            } else {
                shared_data[i * tile_width + j] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    // Compute max pooling from shared memory
    if (x + threadIdx.x < output_width && y + threadIdx.y < output_height) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int shmem_row = threadIdx.y * stride + kh;
                const int shmem_col = threadIdx.x * stride + kw;
                if (shmem_row < tile_height && shmem_col < tile_width) {
                    max_val = max(max_val, shared_data[shmem_row * tile_width + shmem_col]);
                }
            }
        }
        
        const int out_idx = z * (output_height * output_width) + 
                           (y + threadIdx.y) * output_width + 
                           (x + threadIdx.x);
        output[out_idx] = max_val;
    }
}

torch::Tensor maxpool2d_shared_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Optimized block size for H100's 1024 threads/SM capability
    const dim3 block(32, 8);  // 256 threads per block
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        channels * batch_size
    );

    // Calculate shared memory size per block
    const int tile_width = block.x + kernel_size - 1;
    const int tile_height = block.y + kernel_size - 1;
    const size_t shared_mem_size = tile_width * tile_height * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool2d_shared_forward", ([&] {
        maxpool2d_shared_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_height, input_width,
            output_height, output_width,
            kernel_size, stride,
            padding, dilation,
            channels, batch_size
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &maxpool2d_shared_forward, "MaxPool2D with shared memory (CUDA)");
}
