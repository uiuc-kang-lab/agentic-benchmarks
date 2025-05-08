#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Optimized CUDA kernel combining balanced workload and efficient memory access
__global__ void optimized_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    // Determine batch and channel based on blockIdx.z
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Optimized tile dimensions based on typical use cases
    const int tile_x = 32;
    const int tile_y = 4;

    // Compute starting indices for this tile
    int ow_start = blockIdx.x * tile_x;
    int oh_start = blockIdx.y * tile_y;

    // Each thread computes one output element in the tile
    int ow = ow_start + threadIdx.x;
    int oh = oh_start + threadIdx.y;

    // Shared memory for both weights and input data
    extern __shared__ float shared_mem[];
    float* sweight = shared_mem;
    float* sinput = &shared_mem[kernel_h];

    // Load weights into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    __syncthreads();

    if (oh < out_h && ow < out_w) {
        float sum = 0.f;
        
        // Prefetch input data for the entire tile
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_index = ((b * channels + c) * in_h + ih) * in_w + iw;
                float in_val = __ldg(&input[in_index]);
                sum += in_val * sweight[kh];
            }
        }
        
        // Add bias and write output
        sum += __ldg(&bias[c]);
        int out_index = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_index] = sum;
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val = bias.has_value() && bias.value().defined() 
        ? bias.value().contiguous() 
        : at::zeros({channels}, x.options());

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    const int tile_x = 32;
    const int tile_y = 4;
    
    dim3 block(tile_x, tile_y, 1);
    dim3 grid((out_w + tile_x - 1) / tile_x, 
              (out_h + tile_y - 1) / tile_y, 
              batch * channels);

    // Shared memory size for weights and input cache
    int shmem_size = (kernel_h + tile_x * tile_y) * sizeof(float);

    optimized_depthwise_conv2d_kernel<<<grid, block, shmem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}