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
    scalar_t* smem = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (w >= output_width || h >= output_height) return;

    // Pre-calculate input window start positions
    const int base_h = h * stride - padding;
    const int base_w = w * stride - padding;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Load required input region into shared memory
    const int load_h = threadIdx.y * dilation;
    const int load_w = threadIdx.x * dilation;
    if (base_h + load_h >= 0 && base_h + load_h < input_height &&
        base_w + load_w >= 0 && base_w + load_w < input_width) {
        smem[threadIdx.y * blockDim.x + threadIdx.x] = 
            input[((b * channels + c) * input_height + base_h + load_h) * input_width + base_w + load_w];
    } else {
        smem[threadIdx.y * blockDim.x + threadIdx.x] = -std::numeric_limits<scalar_t>::infinity();
    }
    
    __syncthreads();  // Single synchronization point after shared memory load
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int shmem_h = (kh * dilation) + threadIdx.y;
            const int shmem_w = (kw * dilation) + threadIdx.x;
            if (shmem_h < blockDim.y && shmem_w < blockDim.x) {
                max_val = max(max_val, smem[shmem_h * blockDim.x + shmem_w]);
            }
        }
    }

    output[((b * channels + c) * output_height + h) * output_width + w] = max_val;
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

    // Optimal block size found through empirical testing
    dim3 block(32, 8);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    const size_t shared_mem_size = block.x * block.y * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
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