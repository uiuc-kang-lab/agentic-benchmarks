#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Adaptive tile sizes based on kernel size
#define BASE_TILE_W 16
#define BASE_TILE_H 16
#define MAX_KERNEL_SIZE 11
#define MAX_SHARED_MEM (48 * 1024) // 48KB typical max shared memory per block

template <typename scalar_t>
__global__ void adaptiveDepthwiseConv2DKernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int tile_w,
    const int tile_h) {

    // Dynamic shared memory allocation
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);
    
    // Calculate shared memory dimensions including halo regions
    const int shared_width = (tile_w - 1) * stride + kernel_size;
    const int shared_height = (tile_h - 1) * stride + kernel_size;

    // Block and input coordinates
    const int out_tile_x = blockIdx.x * tile_w;
    const int out_tile_y = blockIdx.y * tile_h;
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;
    
    // Thread coordinates
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Input tile coordinates
    const int in_tile_x = out_tile_x * stride - padding;
    const int in_tile_y = out_tile_y * stride - padding;

    // Collaborative loading of input data into shared memory
    #pragma unroll 4
    for (int j = ty; j < shared_height; j += blockDim.y) {
        #pragma unroll 4
        for (int i = tx; i < shared_width; i += blockDim.x) {
            const int in_x = in_tile_x + i;
            const int in_y = in_tile_y + j;
            
            scalar_t val = 0;
            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                const int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
                val = x[input_idx];
            }
            shmem[j * shared_width + i] = val;
        }
    }
    __syncthreads();

    // Process output elements
    if (tx < tile_w && ty < tile_h) {
        const int out_x = out_tile_x + tx;
        const int out_y = out_tile_y + ty;
        
        if (out_x < out_width && out_y < out_height) {
            scalar_t sum = 0;
            const int sh_x = tx * stride;
            const int sh_y = ty * stride;
            
            // Use registers for weight cache if kernel is small
            scalar_t weight_cache[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
            if (kernel_size <= MAX_KERNEL_SIZE) {
                #pragma unroll
                for (int kh = 0; kh < kernel_size; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        weight_cache[kh * kernel_size + kw] = w[(c * kernel_size + kh) * kernel_size + kw];
                    }
                }
            }

            // Compute convolution
            #pragma unroll 4
            for (int kh = 0; kh < kernel_size; ++kh) {
                #pragma unroll 4
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int shared_idx = (sh_y + kh) * shared_width + (sh_x + kw);
                    const scalar_t weight_val = (kernel_size <= MAX_KERNEL_SIZE) ? 
                        weight_cache[kh * kernel_size + kw] : 
                        w[(c * kernel_size + kh) * kernel_size + kw];
                    sum += shmem[shared_idx] * weight_val;
                }
            }
            
            sum += b[c];
            const int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
            out[out_idx] = sum;
        }
    }
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    // Adaptive tile sizing based on kernel size and available shared memory
    int tile_w = BASE_TILE_W;
    int tile_h = BASE_TILE_H;
    
    // Adjust tile size if needed based on kernel size and shared memory constraints
    const size_t elem_size = sizeof(float);
    const size_t required_shmem = ((tile_w - 1) * stride + kernel_size) * 
                                 ((tile_h - 1) * stride + kernel_size) * 
                                 elem_size;
    
    while (required_shmem > MAX_SHARED_MEM) {
        if (tile_w > tile_h) tile_w /= 2;
        else tile_h /= 2;
    }

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    dim3 block(tile_w, tile_h);
    dim3 grid(
        (out_width + tile_w - 1) / tile_w,
        (out_height + tile_h - 1) / tile_h,
        batch_size * in_channels
    );

    const size_t shmem_size = ((tile_w - 1) * stride + kernel_size) * 
                             ((tile_h - 1) * stride + kernel_size) * 
                             sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "adaptive_depthwise_conv2d", ([&] {
        adaptiveDepthwiseConv2DKernel<scalar_t><<<grid, block, shmem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            kernel_size, out_height, out_width,
            stride, padding, tile_w, tile_h
        );
    }));

    return out;
}