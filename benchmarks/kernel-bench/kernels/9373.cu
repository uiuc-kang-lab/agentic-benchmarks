#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHANNELS_PER_BLOCK 4
#define TILE_SIZE 16
#define BLOCK_SIZE 16

// Helper function to do vectorized loads where possible
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_height,
    const int input_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int height_out,
    const int width_out,
    const int stride,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w) {

    // Thread and block index calculation
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output position
    const int h_out = by * BLOCK_SIZE + ty;
    const int w_out = bx * BLOCK_SIZE + tx;

    // Calculate batch and output channel group
    const int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    const int b = bz / groups_per_batch;
    const int g = bz % groups_per_batch;
    const int oc_start = g * CHANNELS_PER_BLOCK;

    // Early exit if outside bounds
    if (h_out >= height_out || w_out >= width_out || b >= batch_size) return;

    // Shared memory for weights - aligned to 128 bytes for better memory access
    extern __shared__ float shared_weight[];

    // Load weights into shared memory
    const int thread_id = ty * BLOCK_SIZE + tx;
    const int total_threads = BLOCK_SIZE * BLOCK_SIZE;
    const int weight_elements = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;

    #pragma unroll 4
    for (int idx = thread_id; idx < weight_elements; idx += total_threads) {
        const int w_oc = idx / (in_channels * kernel_h * kernel_w);
        const int rem = idx % (in_channels * kernel_h * kernel_w);
        const int global_oc = oc_start + w_oc;
        
        if (global_oc < out_channels) {
            shared_weight[idx] = weight[global_oc * in_channels * kernel_h * kernel_w + rem];
        }
    }
    __syncthreads();

    // Pre-compute input base indices
    const int in_batch_offset = b * in_channels * input_height * input_width;
    const int out_batch_offset = b * out_channels * height_out * width_out;

    // Initialize accumulators
    float sums[CHANNELS_PER_BLOCK] = {0.0f};
    if (bias != nullptr) {
        #pragma unroll
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            const int global_oc = oc_start + i;
            if (global_oc < out_channels) {
                sums[i] = bias[global_oc];
            }
        }
    }

    // Main computation loop
    #pragma unroll 2
    for (int ic = 0; ic < in_channels; ic++) {
        const int in_channel_offset = ic * input_height * input_width;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++) {
            const int h_in = h_out * stride + kh * dilation_h - pad_h;
            const bool valid_h = (h_in >= 0 && h_in < input_height);
            
            if (valid_h) {
                const int in_h_offset = h_in * input_width;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int w_in = w_out * stride + kw * dilation_w - pad_w;
                    const bool valid_w = (w_in >= 0 && w_in < input_width);
                    
                    if (valid_w) {
                        const float x_val = __ldg(&x[in_batch_offset + in_channel_offset + in_h_offset + w_in]);
                        
                        #pragma unroll
                        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                            if (oc_start + i < out_channels) {
                                const int weight_idx = i * (in_channels * kernel_h * kernel_w) +
                                                     ic * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;
                                sums[i] += x_val * shared_weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Write results - ensure aligned writes where possible
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        const int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            const int out_idx = out_batch_offset +
                              global_oc * height_out * width_out +
                              h_out * width_out + w_out;
            output[out_idx] = sums[i];
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks((width_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size * groups_per_batch);

    // Ensure shared memory is aligned to 128 bytes
    size_t shared_mem_size = (CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w * sizeof(float) + 127) & ~127;

    conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}