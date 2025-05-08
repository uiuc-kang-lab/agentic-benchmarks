#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SHARED_MEM_SIZE 4096

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv3d_shared_mem_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {

    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int elements_per_block = (total_elements + gridDim.x - 1) / gridDim.x;
    const int start_idx = bid * elements_per_block;
    const int end_idx = min(start_idx + elements_per_block, total_elements);

    for (int idx = start_idx + tid; idx < end_idx; idx += blockDim.x) {
        // Calculate output position
        const int w_out = idx % out_width;
        int tmp = idx / out_width;
        const int h_out = tmp % out_height;
        tmp /= out_height;
        const int d_out = tmp % out_depth;
        tmp /= out_depth;
        const int c_out = tmp % out_channels;
        const int b = tmp / out_channels;

        // Calculate group information
        const int group = c_out / (out_channels / groups);
        const int in_channels_per_group = in_channels / groups;

        float sum = 0.0f;
        
        // Load weights into shared memory for current output channel
        __shared__ float shared_weights[SHARED_MEM_SIZE];
        const int weights_per_thread = (kernel_d * kernel_h * kernel_w + blockDim.x - 1) / blockDim.x;
        
        for (int i = 0; i < weights_per_thread; i++) {
            const int weight_idx = tid + i * blockDim.x;
            if (weight_idx < kernel_d * kernel_h * kernel_w) {
                const int kd = weight_idx / (kernel_h * kernel_w);
                const int kh = (weight_idx / kernel_w) % kernel_h;
                const int kw = weight_idx % kernel_w;
                shared_weights[weight_idx] = weight[
                    ((c_out * in_channels_per_group) * kernel_d + kd) * kernel_h * kernel_w +
                    kh * kernel_w + kw
                ];
            }
        }
        __syncthreads();

        // Compute convolution using shared memory
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            const int in_c = group * in_channels_per_group + ic;
            
            for (int kd = 0; kd < kernel_d; kd++) {
                const int d_in = d_out * stride - padding + kd * dilation;
                if (d_in < 0 || d_in >= in_depth) continue;

                for (int kh = 0; kh < kernel_h; kh++) {
                    const int h_in = h_out * stride - padding + kh * dilation;
                    if (h_in < 0 || h_in >= in_height) continue;

                    for (int kw = 0; kw < kernel_w; kw++) {
                        const int w_in = w_out * stride - padding + kw * dilation;
                        if (w_in < 0 || w_in >= in_width) continue;

                        const int input_idx = ((b * in_channels + in_c) * in_depth + d_in) * in_height * in_width +
                                            h_in * in_width + w_in;
                        const int weight_idx = kd * kernel_h * kernel_w + kh * kernel_w + kw;
                        
                        sum += input[input_idx] * shared_weights[weight_idx];
                    }
                }
            }
        }

        // Warp-level reduction
        sum = warpReduceSum(sum);

        // First thread in warp writes result
        if (lane == 0) {
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            output[idx] = sum;
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
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

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

    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem_size = kernel_d * kernel_h * kernel_w * sizeof(float);

    conv3d_shared_mem_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with shared memory (CUDA)");
}