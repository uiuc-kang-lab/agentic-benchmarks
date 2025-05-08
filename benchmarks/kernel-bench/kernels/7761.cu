#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_D 4
#define TILE_H 4
#define TILE_W 4

template <typename scalar_t>
__global__ void conv3d_shared_tiled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups) 
{
    extern __shared__ __align__(sizeof(float)) unsigned char shared_mem[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* shared_weights = &shared_input[TILE_D*TILE_H*TILE_W*in_channels];

    const int b = blockIdx.z;
    const int oz = blockIdx.y * blockDim.y + threadIdx.y;
    const int oy = blockIdx.x * blockDim.x + threadIdx.x;

    const int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    const int shared_load_size = blockDim.x * blockDim.y * blockDim.z;

    scalar_t result = 0;

    for (int ox_start = 0; ox_start < out_d; ox_start += gridDim.x*TILE_W) {
        const int ox = ox_start + threadIdx.z;
        if (ox >= out_d || oz >= out_channels || oy >= out_h || b >= batch_size) continue;

        __syncthreads();

        // Cooperative loading of input tile into shared memory
        for (int load_idx = tid; load_idx < in_channels * TILE_D * TILE_H * TILE_W; load_idx += shared_load_size) {
            const int cx = load_idx % TILE_W;
            const int cy = (load_idx / TILE_W) % TILE_H;
            const int cz = (load_idx / (TILE_W * TILE_H)) % TILE_D;
            const int cc = load_idx / (TILE_W * TILE_H * TILE_D);

            const int ix = ox * stride - padding + cz * dilation;
            const int iy = oy * stride - padding + cy * dilation;
            const int iz = threadIdx.z * stride - padding + cx * dilation;

            if (ix >= 0 && ix < in_d && iy >= 0 && iy < in_h && iz >= 0 && iz < in_w) {
                shared_input[load_idx] = input[((b * in_channels + cc) * in_d + ix) * in_h * in_w + iy * in_w + iz];
            } else {
                shared_input[load_idx] = 0;
            }
        }

        // Load weights for current output channel
        if (tid < in_channels * kernel_d * kernel_h * kernel_w) {
            shared_weights[tid] = weight[((oz * in_channels) * kernel_d + (tid / (kernel_h * kernel_w))) * kernel_h * kernel_w + (tid % (kernel_h * kernel_w))];
        }

        __syncthreads();

        // Compute convolution using shared memory
        for (int kz = 0; kz < kernel_d; ++kz) {
            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int sidx = ((kz * TILE_D + threadIdx.z) * TILE_H + ky) * TILE_W + kx;
                        result += shared_input[sidx * in_channels + ic] * shared_weights[((ic * kernel_d) + kz) * kernel_h * kernel_w + ky * kernel_w + kx];
                    }
                }
            }
        }

        output[((b * out_channels + oz) * out_d + ox) * out_h * out_w + oy * out_w + threadIdx.x] = result;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) 
{
    // ... (same dimension calculations and checks as previous kernels)
    // Kernel launch parameters optimized for H100
    const dim3 block(TILE_H, TILE_W, TILE_D);
    const dim3 grid(
        (out_h + TILE_H - 1) / TILE_H,
        (out_channels + block.y - 1) / block.y,
        batch_size
    );
    
    const size_t shared_mem_size = (TILE_D*TILE_H*TILE_W*in_channels + in_channels*kernel_d*kernel_h*kernel_w) * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_shared_tiled", ([&] {
        conv3d_shared_tiled_kernel<scalar_t><<<grid, block, shared_mem_size>>>(...);
    }));
    
    // ... (same bias handling as previous kernels)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution with shared memory tiling");
}