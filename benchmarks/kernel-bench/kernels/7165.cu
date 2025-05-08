#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_KERNEL_SIZE 11
#define SHARED_MEM_SIZE 2048

template <typename scalar_t>
__global__ void conv2d_shared_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    __shared__ scalar_t shared_input[SHARED_MEM_SIZE];
    __shared__ scalar_t shared_weight[SHARED_MEM_SIZE];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    const int batch_idx = blockIdx.z;
    const int out_ch = blockIdx.y;
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= out_height * out_width) return;
    
    const int out_y = out_idx / out_width;
    const int out_x = out_idx % out_width;
    
    scalar_t sum = 0.0f;
    
    for (int ic_chunk = 0; ic_chunk < in_channels; ic_chunk += WARP_SIZE) {
        
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            const int in_y = out_y * stride - padding + ky;
            
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                const int in_x = out_x * stride - padding + kx;
                
                if (tid < WARP_SIZE && (ic_chunk + tid) < in_channels) {
                    const int in_idx = ((batch_idx * in_channels + (ic_chunk + tid)) * in_height + in_y) * in_width + in_x;
                    const bool valid = (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width);
                    shared_input[tid * kernel_size * kernel_size + ky * kernel_size + kx] = 
                        valid ? input[in_idx] : 0.0f;
                }
            }
        }
        
        if (tid < WARP_SIZE && (ic_chunk + tid) < in_channels) {
            #pragma unroll
            for (int k = 0; k < kernel_size * kernel_size; k++) {
                const int weight_idx = (out_ch * in_channels + (ic_chunk + tid)) * kernel_size * kernel_size + k;
                shared_weight[tid * kernel_size * kernel_size + k] = weight[weight_idx];
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int ic = 0; ic < min(WARP_SIZE, in_channels - ic_chunk); ic++) {
            #pragma unroll
            for (int k = 0; k < kernel_size * kernel_size; k++) {
                sum += shared_input[ic * kernel_size * kernel_size + k] *
                       shared_weight[ic * kernel_size * kernel_size + k];
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        const int out_idx = (batch_idx * out_channels + out_ch) * out_height * out_width + out_y * out_width + out_x;
        output[out_idx] = sum + (bias ? bias[out_ch] : 0.0f);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    
    TORCH_CHECK(dilation == 1, "Dilation > 1 not implemented in this kernel");
    TORCH_CHECK(groups == 1, "Groups > 1 not implemented in this kernel");
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks_x = (out_height * out_width + threads - 1) / threads;
    const dim3 blocks(blocks_x, out_channels, batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv2d_shared_kernel", ([&] {
        conv2d_shared_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kernel_size,
            stride,
            padding);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA convolution with shared memory and warp reduction");
}