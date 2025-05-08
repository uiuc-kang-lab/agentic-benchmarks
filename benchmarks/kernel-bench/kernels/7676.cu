#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE_Z 4

template <typename scalar_t>
__global__ void conv3d_shared_memory_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
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

    __shared__ scalar_t shared_input[TILE_SIZE * TILE_SIZE * BLOCK_SIZE_Z];
    __shared__ scalar_t shared_weight[TILE_SIZE * TILE_SIZE * BLOCK_SIZE_Z];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int x_out = blockIdx.x * blockDim.x + tx;
    const int y_out = blockIdx.y * blockDim.y + ty;
    const int z_out = blockIdx.z * blockDim.z + tz;
    
    const int channels_per_group = in_channels / groups;
    
    if (x_out >= out_width || y_out >= out_height || z_out >= out_depth)
        return;

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            const int group = oc / (out_channels / groups);
            scalar_t sum = 0.0f;
            
            for (int ic_block = 0; ic_block < channels_per_group; ic_block += TILE_SIZE) {
                const int ic = group * channels_per_group + ic_block;
                
                const int shared_idx = tz * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;
                if (tx < TILE_SIZE && ty < TILE_SIZE && tz < BLOCK_SIZE_Z && (ic + shared_idx) < in_channels) {
                    const int x_in = x_out * stride - padding;
                    const int y_in = y_out * stride - padding;
                    const int z_in = z_out * stride - padding;
                    
                    if (x_in >= 0 && x_in < in_width && 
                        y_in >= 0 && y_in < in_height && 
                        z_in >= 0 && z_in < in_depth) {
                        shared_input[shared_idx] = input[
                            ((b * in_channels + ic + shared_idx) * in_depth + z_in) * 
                            in_height * in_width + y_in * in_width + x_in];
                    } else {
                        shared_input[shared_idx] = 0;
                    }
                }
                
                if (tx < kernel_w && ty < kernel_h && tz < kernel_d) {
                    const int weight_idx = tz * kernel_h * kernel_w + ty * kernel_w + tx;
                    if ((ic + weight_idx) < in_channels) {
                        shared_weight[weight_idx] = weight[
                            ((oc * channels_per_group + weight_idx) * kernel_d + tz) * 
                            kernel_h * kernel_w + ty * kernel_w + tx];
                    }
                }
                
                __syncthreads();

                for (int kd = 0; kd < kernel_d; kd++) {
                    const int z_in = z_out * stride - padding + kd * dilation;
                    if (z_in < 0 || z_in >= in_depth) continue;
                    
                    #pragma unroll
                    for (int kh = 0; kh < kernel_h; kh++) {
                        const int y_in = y_out * stride - padding + kh * dilation;
                        if (y_in < 0 || y_in >= in_height) continue;
                        
                        #pragma unroll
                        for (int kw = 0; kw < kernel_w; kw++) {
                            const int x_in = x_out * stride - padding + kw * dilation;
                            if (x_in < 0 || x_in >= in_width) continue;
                            
                            const int input_idx = (kd * TILE_SIZE + kh) * TILE_SIZE + kw;
                            const int weight_idx = (kd * kernel_h + kh) * kernel_w + kw;
                            
                            sum += shared_input[input_idx] * shared_weight[weight_idx];
                        }
                    }
                }
                
                __syncthreads();
            }
            
            if (x_out < out_width && y_out < out_height && z_out < out_depth) {
                const int out_idx = ((b * out_channels + oc) * out_depth + z_out) * 
                                   out_height * out_width + y_out * out_width + x_out;
                output[out_idx] = sum;
                
                if (bias != nullptr) {
                    output[out_idx] += bias[oc];
                }
            }
        }
    }
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
    
    dim3 threads(TILE_SIZE, TILE_SIZE, BLOCK_SIZE_Z);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        (out_depth + threads.z - 1) / threads.z
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_shared_memory_kernel", ([&] {
        conv3d_shared_memory_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            out_depth, out_height, out_width,
            stride, padding, dilation, groups
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with shared memory optimization (CUDA)");
}