#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define ELEMENTS_PER_THREAD 4
#define MAX_KERNEL_SIZE 7  // Adjust based on expected maximum kernel size

__constant__ float c_weight[1024];  // Constant memory for weights

__global__ void combined_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group
) {
    const int b = blockIdx.z / out_channels;
    const int c = blockIdx.z % out_channels;
    const int g = c / channels_per_group;
    const int m = c % channels_per_group;

    // Shared memory tile dimensions
    const int tile_out_width = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    const int shared_tile_height = BLOCK_SIZE * stride_h + (kernel_h - 1) * dilation_h;

    extern __shared__ float shared_input[];

    // Load weights into registers for faster access
    float w_cache[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    #pragma unroll
    for (int i = 0; i < kernel_h * kernel_w; ++i) {
        w_cache[i] = weight[((g * channels_per_group + m) * kernel_h * kernel_w) + i];
    }

    // Thread indices
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int thread_id = tidy * blockDim.x + tidx;
    const int total_threads = blockDim.x * blockDim.y;

    // Input tile loading coordinates
    const int base_in_x = blockIdx.x * tile_out_width * stride_w - padding_w;
    const int base_in_y = blockIdx.y * BLOCK_SIZE * stride_h - padding_h;

    // Efficient shared memory loading using vectorized loads when possible
    const int shared_size = shared_tile_width * shared_tile_height;
    const int input_batch_offset = (b * in_channels + g) * in_h * in_w;
    
    #pragma unroll 4
    for (int idx = thread_id; idx < shared_size; idx += total_threads) {
        const int sh_y = idx / shared_tile_width;
        const int sh_x = idx % shared_tile_width;
        const int in_y = base_in_y + sh_y;
        const int in_x = base_in_x + sh_x;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
            val = input[input_batch_offset + in_y * in_w + in_x];
        }
        shared_input[idx] = val;
    }
    __syncthreads();

    // Output coordinates
    const int out_y = blockIdx.y * BLOCK_SIZE + tidy;
    if (out_y >= out_h) return;

    const int base_out_x = blockIdx.x * tile_out_width + tidx * ELEMENTS_PER_THREAD;
    
    // Pre-compute shared memory base indices
    const int sh_y_base = tidy * stride_h;
    const int sh_x_base = tidx * ELEMENTS_PER_THREAD * stride_w;

    // Result accumulators
    float results[ELEMENTS_PER_THREAD] = {0.0f};

    // Manual unroll for 3x3 kernel (common case)
    if (kernel_h == 3 && kernel_w == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int sh_y = sh_y_base + kh * dilation_h;
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const float w_val = w_cache[kh * 3 + kw];
                const int kw_offset = kw * dilation_w;
                
                #pragma unroll
                for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
                    const int sh_x = sh_x_base + v * stride_w + kw_offset;
                    results[v] += shared_input[sh_y * shared_tile_width + sh_x] * w_val;
                }
            }
        }
    }
    // General case
    else {
        #pragma unroll 2
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int sh_y = sh_y_base + kh * dilation_h;
            #pragma unroll 2
            for (int kw = 0; kw < kernel_w; ++kw) {
                const float w_val = w_cache[kh * kernel_w + kw];
                const int kw_offset = kw * dilation_w;
                
                #pragma unroll
                for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
                    const int sh_x = sh_x_base + v * stride_w + kw_offset;
                    results[v] += shared_input[sh_y * shared_tile_width + sh_x] * w_val;
                }
            }
        }
    }

    // Write results to global memory
    const float bias_val = bias != nullptr ? bias[c] : 0.0f;
    const int out_batch_offset = ((b * out_channels + c) * out_h + out_y) * out_w;

    #pragma unroll
    for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
        const int out_x = base_out_x + v;
        if (out_x < out_w) {
            output[out_batch_offset + out_x] = results[v] + bias_val;
        }
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

    // Copy weights to constant memory
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_w + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD),
        (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );

    int shared_tile_width = (BLOCK_SIZE * ELEMENTS_PER_THREAD) * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = BLOCK_SIZE * stride_h + (kernel_h - 1) * dilation_h;
    int shared_mem_size = shared_tile_width * shared_tile_height * sizeof(float);

    combined_depthwise_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Depthwise Conv2D forward (CUDA)");
}