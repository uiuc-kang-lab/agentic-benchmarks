#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4
#define WARPS_PER_BLOCK ((BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z) / WARP_SIZE)

template<typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename scalar_t>
__global__ void conv3d_shared_memory_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    const int batch_size, const int in_channels, const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int out_depth, const int out_height, const int out_width,
    const int stride, const int padding, const int dilation,
    const int groups) {

    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int d_out = blockIdx.x * BLOCK_SIZE_Z + tz;
    const int h_out = blockIdx.y * BLOCK_SIZE_Y + ty;
    const int w_out = blockIdx.z * BLOCK_SIZE_X + tx;
    
    const int lane_id = threadIdx.x + threadIdx.y * BLOCK_SIZE_X + threadIdx.z * (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    const int warp_id = lane_id / WARP_SIZE;
    const int lane_in_warp = lane_id % WARP_SIZE;

    if (d_out >= out_depth || h_out >= out_height || w_out >= out_width) return;

    const int channels_per_group = in_channels / groups;
    
    for (int b = 0; b < batch_size; b++) {
        for (int oc = warp_id; oc < out_channels; oc += WARPS_PER_BLOCK) {
            const int group = oc / (out_channels / groups);
            scalar_t sum = 0.0f;

            for (int ic = 0; ic < channels_per_group; ic++) {
                const int in_c = group * channels_per_group + ic;
                
                // Load input patch into shared memory
                for (int kd = 0; kd < kernel_d; kd++) {
                    const int d_in = d_out * stride - padding + kd * dilation;
                    if (d_in >= 0 && d_in < in_depth) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            const int h_in = h_out * stride - padding + kh * dilation;
                            if (h_in >= 0 && h_in < in_height) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    const int w_in = w_out * stride - padding + kw * dilation;
                                    if (w_in >= 0 && w_in < in_width) {
                                        const int shared_idx = ((kd * kernel_h + kh) * kernel_w + kw) * WARP_SIZE + lane_in_warp;
                                        const int input_idx = ((b * in_channels + in_c) * in_depth + d_in) * in_height * in_width +
                                                            h_in * in_width + w_in;
                                        if (shared_idx < kernel_d * kernel_h * kernel_w * WARP_SIZE) {
                                            shared_input[shared_idx] = input[input_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                __syncwarp();

                // Compute convolution using shared memory
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            const int d_in = d_out * stride - padding + kd * dilation;
                            const int h_in = h_out * stride - padding + kh * dilation;
                            const int w_in = w_out * stride - padding + kw * dilation;
                            
                            if (d_in >= 0 && d_in < in_depth &&
                                h_in >= 0 && h_in < in_height &&
                                w_in >= 0 && w_in < in_width) {
                                const int shared_idx = ((kd * kernel_h + kh) * kernel_w + kw) * WARP_SIZE + lane_in_warp;
                                const int weight_idx = ((oc * channels_per_group + ic) * kernel_d + kd) * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;
                                sum += shared_input[shared_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                __syncwarp();
            }

            // Warp-level reduction
            sum = warpReduceSum(sum);

            // First thread in warp writes result
            if (lane_in_warp == 0) {
                const int out_idx = ((b * out_channels + oc) * out_depth + d_out) * out_height * out_width +
                                  h_out * out_width + w_out;
                output[out_idx] = sum + (bias ? bias[oc] : 0.0f);
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
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 blocks(
        (out_depth + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z,
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X
    );

    const int shared_memory_size = kernel_d * kernel_h * kernel_w * WARP_SIZE * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_shared_memory_kernel", ([&] {
        conv3d_shared_memory_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &forward, "3D convolution forward with shared memory and warp reduction (CUDA)");
}