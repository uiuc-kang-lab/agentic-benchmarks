#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block configuration for the 2D kernel branch
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8

// Threshold to decide kernel branch usage based on kernel area
#define SMALL_KERNEL_THRESHOLD 25
// Block size for the shared memory kernel (must be a multiple of warpSize)
#define SHARED_BLOCK_SIZE 64

// Kernel for small convolution kernels using a 2D grid (no shared memory reduction)
__global__ void depthwise_conv2d_kernel_2d(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int b = blockIdx.z / out_channels;
    int c_out = blockIdx.z % out_channels;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_in = y * stride_h - padding_h + kh * dilation_h;
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_in = x * stride_w - padding_w + kw * dilation_w;
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[((b * out_channels + c_out) * out_h + y) * out_w + x] = sum;
}

// Optimized shared memory kernel for larger convolution kernels
// that minimizes __syncthreads() usage by using warp shuffles
__global__ void depthwise_conv2d_kernel_shared_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    int tid = threadIdx.x;
    int index = blockIdx.x; // each block computes one output pixel
    
    if (index >= batch_size * out_channels * out_h * out_w) return;
    
    // Decode output coordinate from index
    int w_out = index % out_w;
    int h_out = (index / out_w) % out_h;
    int c_out = (index / (out_w * out_h)) % out_channels;
    int b = index / (out_w * out_h * out_channels);
    
    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;
    
    int kernel_size = kernel_h * kernel_w;
    float partial_sum = 0.0f;
    // Each thread computes a partial sum over kernel elements
    for (int k = tid; k < kernel_size; k += blockDim.x) {
        int kh = k / kernel_w;
        int kw = k % kernel_w;
        int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        int w_in = w_out * stride_w - padding_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }
    
    // Warp-level reduction using shuffle intrinsics (no __syncthreads needed within a warp)
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }
    
    // Each warp's lane 0 stores its result in shared memory
    __shared__ float warp_sums[SHARED_BLOCK_SIZE / 32];
    int lane = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    if (lane == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    
    // Synchronize once to ensure all warp results are available
    __syncthreads();
    
    // Let the first few threads (one per warp) reduce the warp sums
    float total = 0.0f;
    int num_warps = blockDim.x / warpSize;
    if (tid < num_warps) {
        total = warp_sums[tid];
    }
    
    // Use warp shuffle to reduce the warp sums; only threads in first warp participate
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        total += __shfl_down_sync(mask, total, offset);
    }
    
    if (tid == 0) {
        if (bias != nullptr) {
            total += bias[c_out];
        }
        output[index] = total;
    }
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;
    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    
    // Choose kernel branch based on kernel size
    if (kernel_h * kernel_w <= SMALL_KERNEL_THRESHOLD) {
        dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 blocks(
            (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
            (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
            batch_size * out_channels
        );

        depthwise_conv2d_kernel_2d<<<blocks, threads>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            channels_per_group
        );
    } else {
        // For larger kernel sizes, use the optimized shared memory reduction kernel
        int total_outputs = batch_size * out_channels * out_h * out_w;
        int blockSize = SHARED_BLOCK_SIZE;
        int gridSize = total_outputs;
        depthwise_conv2d_kernel_shared_opt<<<gridSize, blockSize>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            channels_per_group
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Depthwise Conv2D forward with optimized synchronization (CUDA)");
}
