#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define TOUT_W 16
#define TOUT_H 16
#define THRESHOLD_SIZE 8

template <typename scalar_t>
__global__ void depthwiseConv2DKernelSimple(
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
    const int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * out_height * out_width) return;

    int w_out_idx = idx % out_width;
    int tmp = idx / out_width;
    int h_out_idx = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    scalar_t value = 0;
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out_idx * stride - padding + kh;
            int w_in = w_out_idx * stride - padding + kw;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                value += x[((n * in_channels + c) * in_height + h_in) * in_width + w_in] * 
                        w[((c * 1 + 0) * kernel_size + kh) * kernel_size + kw];
            }
        }
    }
    out[idx] = value + b[c];
}

template <typename scalar_t>
__global__ void depthwiseConv2DKernelTiled(
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
    const int padding) {
    int nc = blockIdx.z;
    int c = nc % in_channels;
    int n = nc / in_channels;
    
    int tile_out_x = blockIdx.x * TOUT_W;
    int tile_out_y = blockIdx.y * TOUT_H;
    
    extern __shared__ char smem[];
    scalar_t* smem_weight = reinterpret_cast<scalar_t*>(smem);
    scalar_t* smem_input = smem_weight + (kernel_size * kernel_size);
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    for (int i = tid; i < kernel_size * kernel_size; i += blockDim.x * blockDim.y) {
        smem_weight[i] = w[c * kernel_size * kernel_size + i];
    }
    
    int tile_in_x = tile_out_x * stride - padding;
    int tile_in_y = tile_out_y * stride - padding;
    int tile_in_w = (TOUT_W - 1) * stride + kernel_size;
    int tile_in_h = (TOUT_H - 1) * stride + kernel_size;
    
    for (int i = tid; i < tile_in_w * tile_in_h; i += blockDim.x * blockDim.y) {
        int tx = i % tile_in_w;
        int ty = i / tile_in_w;
        int in_x = tile_in_x + tx;
        int in_y = tile_in_y + ty;
        scalar_t val = 0;
        if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
            val = x[((n * in_channels + c) * in_height + in_y) * in_width + in_x];
        }
        smem_input[ty * tile_in_w + tx] = val;
    }
    __syncthreads();
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = tile_out_x + tx;
    int out_y = tile_out_y + ty;
    
    if (tx < TOUT_W && ty < TOUT_H && out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        #pragma unroll
        for (int i = 0; i < kernel_size; i++) {
            #pragma unroll
            for (int j = 0; j < kernel_size; j++) {
                sum += smem_input[(ty * stride + i) * tile_in_w + (tx * stride + j)] * 
                       smem_weight[i * kernel_size + j];
            }
        }
        sum += b[c];
        out[((n * in_channels + c) * out_height + out_y) * out_width + out_x] = sum;
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
    
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d", ([&] {
        if (kernel_size <= THRESHOLD_SIZE) {
            const int total = batch_size * in_channels * out_height * out_width;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            
            depthwiseConv2DKernelSimple<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), 
                bias.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                batch_size, in_channels, in_height, in_width,
                kernel_size, out_height, out_width, stride, padding);
        } else {
            dim3 block(TOUT_W, TOUT_H);
            dim3 grid((out_width + TOUT_W - 1) / TOUT_W,
                     (out_height + TOUT_H - 1) / TOUT_H,
                     batch_size * in_channels);
                     
            int tile_in_w = (TOUT_W - 1) * stride + kernel_size;
            int tile_in_h = (TOUT_H - 1) * stride + kernel_size;
            size_t smem_size = (kernel_size * kernel_size + tile_in_w * tile_in_h) * sizeof(scalar_t);
            
            depthwiseConv2DKernelTiled<scalar_t><<<grid, block, smem_size>>>(
                x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                batch_size, in_channels, in_height, in_width,
                kernel_size, out_height, out_width, stride, padding);
        }
    }));
    
    return out;
}