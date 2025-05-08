#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Warp-level reduction function
template<typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Combined optimized 3D convolution kernel using grid-stride and warp-level primitives
__global__ void conv3d_opt_kernel(
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
    const int groups
) {
    // Warp and lane indices
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int num_warps = warps_per_block * gridDim.x;
    
    // Total number of output elements
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    // Grid-stride loop distributed at warp granularity
    for (int idx = blockIdx.x * warps_per_block + warp_id; idx < total_elements; idx += num_warps) {
        // Decode the linear index into multi-dimensional indices
        int tmp = idx;
        const int w_out = tmp % out_width;
        tmp /= out_width;
        const int h_out = tmp % out_height;
        tmp /= out_height;
        const int d_out = tmp % out_depth;
        tmp /= out_depth;
        const int c_out = tmp % out_channels;
        const int b = tmp / out_channels;

        float sum = 0.0f;

        // Determine group and channel index within the group
        const int group = c_out / (out_channels / groups);
        const int in_channels_per_group = in_channels / groups;

        // Loop over input channels in group
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            const int in_c = group * in_channels_per_group + ic;
            // Distribute kernel volume among warp lanes
            for (int k_idx = lane_id; k_idx < kernel_d * kernel_h * kernel_w; k_idx += WARP_SIZE) {
                int kd = k_idx / (kernel_h * kernel_w);
                int rem = k_idx % (kernel_h * kernel_w);
                int kh = rem / kernel_w;
                int kw = rem % kernel_w;

                int d_in = d_out * stride - padding + kd * dilation;
                int h_in = h_out * stride - padding + kh * dilation;
                int w_in = w_out * stride - padding + kw * dilation;

                // Check bounds
                if (d_in >= 0 && d_in < in_depth &&
                    h_in >= 0 && h_in < in_height &&
                    w_in >= 0 && w_in < in_width) {
                    
                    int input_idx = ((b * in_channels + in_c) * in_depth + d_in) * (in_height * in_width) +
                                    h_in * in_width + w_in;
                    int weight_idx = ((c_out * in_channels_per_group + ic) * kernel_d + kd) * (kernel_h * kernel_w) +
                                     kh * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        // Warp-level reduction across lanes
        sum = warpReduceSum(sum);
        
        // First lane writes the final result for this output element
        if (lane_id == 0) {
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            output[idx] = sum;
        }
    }
}

// Host function to launch the kernel
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
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Weight dimensions
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    // Launch using BLOCK_SIZE threads per block
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    conv3d_opt_kernel<<<num_blocks, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Optimized 3D convolution forward (CUDA) combining warp and grid-stride loops");
}
