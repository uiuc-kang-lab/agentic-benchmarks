#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_DIM 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void uniform_depthwise_conv2d_kernel(
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
    
    // Block and thread indexing
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int batch_channel = blockIdx.z;
    const int b = batch_channel / channels;
    const int c = batch_channel % channels;
    
    // Early exit for invalid batch indices
    if (b >= batch) return;
    
    // Calculate tile boundaries
    const int tile_start_y = blockIdx.y * TILE_DIM;
    const int tile_start_x = blockIdx.x * TILE_DIM;
    
    // Pre-compute channel offset for input and output
    const int channel_offset = (b * channels + c);
    const float bias_val = bias[c];
    
    // Each warp processes a row within the tile
    const int row = tile_start_y + warp_id;
    if (row < out_h) {
        // Pre-compute row-specific values
        const int base_ih = row * stride - padding;
        
        // Process elements within the row using warp lanes
        #pragma unroll 4
        for (int col = tile_start_x + lane_id; col < min(tile_start_x + TILE_DIM, out_w); col += WARP_SIZE) {
            const int base_iw = col * stride - padding;
            float sum = 0.0f;
            
            // Compute convolution for valid input positions only
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ih = base_ih + kh * dilation;
                const int iw = base_iw;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    const int input_idx = (channel_offset * in_h + ih) * in_w + iw;
                    const int weight_idx = c * kernel_h + kh;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
            
            // Write output
            const int output_idx = (channel_offset * out_h + row) * out_w + col;
            output[output_idx] = sum + bias_val;
        }
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
        throw std::invalid_argument("Depthwise convolution requires groups == channels");
    }
    
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }
    
    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - 1) / stride + 1;
    
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());
    
    dim3 grid(
        (out_w + TILE_DIM - 1) / TILE_DIM,
        (out_h + TILE_DIM - 1) / TILE_DIM,
        batch * channels
    );
    dim3 block(BLOCK_SIZE);
    
    uniform_depthwise_conv2d_kernel<<<grid, block>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}