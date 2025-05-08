#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Device function to load weights into shared memory with vectorized loading
__device__ __forceinline__ void load_weights_to_shared(
    const float* __restrict__ weight,
    float* sweight,
    int c,
    int kernel_h) {
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Use vector loads for better memory bandwidth utilization
    if (tid * 4 < kernel_h) {
        float4* sweight_vec = reinterpret_cast<float4*>(&sweight[tid * 4]);
        const float4* weight_vec = reinterpret_cast<const float4*>(&weight[c * kernel_h + tid * 4]);
        if (tid * 4 + 4 <= kernel_h) {
            *sweight_vec = *weight_vec;
        } else {
            // Handle remaining elements
            for (int i = 0; i < kernel_h - tid * 4; ++i) {
                sweight[tid * 4 + i] = weight[c * kernel_h + tid * 4 + i];
            }
        }
    }
    __syncthreads();
}

// Optimized kernel combining best practices from both implementations
__global__ void optimized_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int stride,
    const int padding,
    const int dilation) {

    // Determine batch and channel based on blockIdx.z
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    // Use 32x8 tile size for better occupancy
    const int tile_x = 32;
    const int tile_y = 8;

    // Compute starting indices for this tile
    const int ow_start = blockIdx.x * tile_x;
    const int oh_start = blockIdx.y * tile_y;

    // Calculate thread position
    const int ow = ow_start + threadIdx.x;
    const int oh = oh_start + threadIdx.y;

    // Shared memory for weights
    extern __shared__ float sweight[];
    load_weights_to_shared(weight, sweight, c, kernel_h);

    // Pre-compute channel offset for better performance
    const int channel_offset = (b * channels + c);
    const float bias_val = __ldg(&bias[c]);

    if (oh < out_h && ow < out_w) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int ih = oh * stride - padding + kh * dilation;
            const int iw = ow * stride - padding;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                const int in_idx = (channel_offset * in_h + ih) * in_w + iw;
                sum = __fmaf_rn(__ldg(&input[in_idx]), sweight[kh], sum);
            }
        }
        
        const int out_idx = (channel_offset * out_h + oh) * out_w + ow;
        output[out_idx] = sum + bias_val;
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

    const int batch = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val = bias.has_value() && bias.value().defined() 
        ? bias.value().contiguous() 
        : at::zeros({channels}, x.options());

    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    const int tile_x = 32;
    const int tile_y = 8;
    
    dim3 block(tile_x, tile_y, 1);
    dim3 grid((out_w + tile_x - 1) / tile_x, 
              (out_h + tile_y - 1) / tile_y, 
              batch * channels);

    const int shmem_size = kernel_h * sizeof(float);

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