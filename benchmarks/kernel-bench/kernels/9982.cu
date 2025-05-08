#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macro definitions
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8

// For small kernels, use non-atomic kernel
// For large kernels, if kernel_size > ATOMIC_KERNEL_THRESHOLD then use atomic-based multi-block kernel
#define ATOMIC_KERNEL_THRESHOLD 64

// For atomic kernel, use a fixed number of blocks per output pixel
#define ATOMIC_BLOCKS_PER_OUTPUT 4
#define ATOMIC_BLOCK_SIZE 128

// -----------------------------------------------------------------------------
// Kernel for small convolution kernels (non-atomic version) using 2D block organization
// Each thread computes one output pixel without race conditions
// -----------------------------------------------------------------------------
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

    if (x >= out_w || y >= out_h || b >= batch_size)
        return;

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

// -----------------------------------------------------------------------------
// Atomic Kernel for large convolution kernels using multi-block accumulation
// Each output pixel is computed by ATOMIC_BLOCKS_PER_OUTPUT blocks.
// Each block computes a partial sum over a partition of the kernel elements and does a reduction via shared memory.
// Then, thread 0 performs an atomicAdd to the global output. Only the block with partition index 0 adds the bias.
// -----------------------------------------------------------------------------
__global__ void depthwise_conv2d_kernel_atomic(
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
    int channels_per_group,
    int kernel_size,
    int partition_size  // number of kernel elements per partition
) {
    // Grid: blockIdx.x in [0, ATOMIC_BLOCKS_PER_OUTPUT) is the partition index
    //       blockIdx.y in [0, total_out), where total_out = batch_size*out_channels*out_h*out_w
    int out_index = blockIdx.y;
    int part = blockIdx.x;  // partition index

    // Decode output index: out_index corresponds to output[b, c_out, h_out, w_out]
    int w_out = out_index % out_w;
    int h_out = (out_index / out_w) % out_h;
    int c_out = (out_index / (out_w * out_h)) % out_channels;
    int b = out_index / (out_w * out_h * out_channels);

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    // Determine the range of kernel indices for this partition
    int start_k = part * partition_size;
    int end_k = start_k + partition_size;
    if (end_k > kernel_size)
        end_k = kernel_size;

    float sum = 0.0f;
    // Loop over assigned kernel indices
    for (int k = start_k + threadIdx.x; k < end_k; k += blockDim.x) {
        int kh = k / kernel_w;
        int kw = k % kernel_w;
        int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        int w_in = w_out * stride_w - padding_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Reduction within block using shared memory
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float block_sum = sdata[0];

    // Thread 0 of each block writes its partial sum to global memory using atomicAdd
    if (tid == 0) {
        // If this is the first partition, add the bias (if provided) exactly once
        if (part == 0 && bias != nullptr) {
            block_sum += bias[c_out];
        }
        atomicAdd(&output[out_index], block_sum);
    }
}

// -----------------------------------------------------------------------------
// Forward function - selects kernel variant based on kernel size
// -----------------------------------------------------------------------------

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
    int kernel_size = kernel_h * kernel_w;

    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    torch::Tensor output;
    
    // For small kernels, use the non-atomic 2D kernel
    if (kernel_size <= ATOMIC_KERNEL_THRESHOLD) {
        // Allocate output without initialization
        output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
        
        dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 blocks(
            (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
            (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
            batch_size * out_channels
        );

        depthwise_conv2d_kernel_2d<<<blocks, threads>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias->data_ptr<float>() : nullptr,
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
        // For large kernels, use the atomic kernel which splits the work across multiple blocks per output pixel.
        // Initialize output to zero because atomicAdd accumulates on it.
        output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

        // Define grid dimensions for atomic kernel
        int total_out = batch_size * out_channels * out_h * out_w;
        dim3 grid(ATOMIC_BLOCKS_PER_OUTPUT, total_out);
        dim3 block(ATOMIC_BLOCK_SIZE);

        // Determine partition size: number of kernel elements per block partition
        int partition_size = (kernel_size + ATOMIC_BLOCKS_PER_OUTPUT - 1) / ATOMIC_BLOCKS_PER_OUTPUT;

        // Launch the atomic kernel. Shared memory size: ATOMIC_BLOCK_SIZE * sizeof(float)
        depthwise_conv2d_kernel_atomic<<<grid, block, ATOMIC_BLOCK_SIZE * sizeof(float)>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias->data_ptr<float>() : nullptr,
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
            channels_per_group,
            kernel_size,
            partition_size
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward with minimal atomics (CUDA)");
}
