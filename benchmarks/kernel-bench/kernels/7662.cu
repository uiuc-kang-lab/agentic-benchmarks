#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4
#define WARP_SIZE 32

template<typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv3d_shared_memory_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

    extern __shared__ float shared_mem[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int d_out = blockIdx.z * BLOCK_SIZE_Z + tz;
    const int h_out = blockIdx.y * BLOCK_SIZE_Y + ty;
    const int w_out = blockIdx.x * BLOCK_SIZE_X + tx;
    
    const int lane_id = threadIdx.x + threadIdx.y * BLOCK_SIZE_X + threadIdx.z * BLOCK_SIZE_X * BLOCK_SIZE_Y;
    const int warp_id = lane_id / WARP_SIZE;
    const int lane_in_warp = lane_id % WARP_SIZE;
    
    const int channels_per_group = out_channels / groups;
    
    if (d_out < out_depth && h_out < out_height && w_out < out_width) {
        for (int b = 0; b < batch_size; b++) {
            for (int g = 0; g < groups; g++) {
                for (int oc = g * channels_per_group; oc < (g + 1) * channels_per_group; oc++) {
                    float sum = 0.0f;
                    
                    const int in_channels_per_group = in_channels / groups;
                    const int in_channel_start = g * in_channels_per_group;
                    const int in_channel_end = (g + 1) * in_channels_per_group;
                    
                    // Load kernel weights into shared memory
                    for (int load_idx = lane_id; load_idx < kernel_d * kernel_h * kernel_w * in_channels_per_group; load_idx += BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z) {
                        const int kd = load_idx / (kernel_h * kernel_w * in_channels_per_group);
                        const int remaining = load_idx % (kernel_h * kernel_w * in_channels_per_group);
                        const int kh = remaining / (kernel_w * in_channels_per_group);
                        const int remaining2 = remaining % (kernel_w * in_channels_per_group);
                        const int kw = remaining2 / in_channels_per_group;
                        const int ic = remaining2 % in_channels_per_group;
                        
                        shared_mem[load_idx] = weight[((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h * kernel_w + 
                                                    kh * kernel_w + kw];
                    }
                    __syncthreads();
                    
                    // Compute convolution using shared memory
                    for (int ic = 0; ic < in_channels_per_group; ic++) {
                        const int in_c = in_channel_start + ic;
                        
                        for (int kd = 0; kd < kernel_d; kd++) {
                            const int d_in = d_out * stride - padding + kd * dilation;
                            if (d_in < 0 || d_in >= in_depth) continue;
                            
                            for (int kh = 0; kh < kernel_h; kh++) {
                                const int h_in = h_out * stride - padding + kh * dilation;
                                if (h_in < 0 || h_in >= in_height) continue;
                                
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    const int w_in = w_out * stride - padding + kw * dilation;
                                    if (w_in < 0 || w_in >= in_width) continue;
                                    
                                    const float input_val = input[((b * in_channels + in_c) * in_depth + d_in) * in_height * in_width +
                                                                 h_in * in_width + w_in];
                                    const int weight_idx = (kd * kernel_h * kernel_w + kh * kernel_w + kw) * in_channels_per_group + ic;
                                    sum += input_val * shared_mem[weight_idx];
                                }
                            }
                        }
                    }
                    
                    // Warp-level reduction
                    sum = warpReduceSum(sum);
                    
                    // First thread in warp writes result
                    if (lane_in_warp == 0) {
                        float final_sum = sum;
                        if (bias != nullptr) {
                            final_sum += bias[oc];
                        }
                        
                        const int out_idx = ((b * out_channels + oc) * out_depth + d_out) * out_height * out_width +
                                          h_out * out_width + w_out;
                        output[out_idx] = final_sum;
                    }
                }
            }
        }
    }
    __syncthreads();
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    
    auto bias = bias_opt.value_or(at::Tensor());
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 blocks(
        (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_depth + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );
    
    const int shared_mem_size = kernel_d * kernel_h * kernel_w * (in_channels / groups) * sizeof(float);
    
    conv3d_shared_memory_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with shared memory optimization (CUDA)");
}