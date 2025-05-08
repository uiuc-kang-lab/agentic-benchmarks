#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define CHANNELS_PER_BLOCK 4
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;
    float* shared_partial_sums = &shared_mem[CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int h_out = by * BLOCK_SIZE_Y + ty;
    const int w_out = bx * BLOCK_SIZE_X + tx;
    
    const int lane_id = tx % WARP_SIZE;
    const int warp_id = (ty * BLOCK_SIZE_X + tx) / WARP_SIZE;
    const int num_warps = (BLOCK_SIZE_X * BLOCK_SIZE_Y) / WARP_SIZE;

    const int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    const int b = bz / groups_per_batch;
    const int g = bz % groups_per_batch;
    const int oc_start = g * CHANNELS_PER_BLOCK;

    if (h_out >= height_out || w_out >= width_out || b >= batch_size) return;

    // Load weights into shared memory
    const int thread_id = ty * BLOCK_SIZE_X + tx;
    const int total_threads = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    const int weight_elements = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    
    for (int i = thread_id; i < weight_elements; i += total_threads) {
        const int w_oc = i / (in_channels * kernel_h * kernel_w);
        const int rem = i % (in_channels * kernel_h * kernel_w);
        const int global_oc = oc_start + w_oc;
        shared_weight[i] = (global_oc < out_channels) ? 
            weight[global_oc * in_channels * kernel_h * kernel_w + rem] : 0.0f;
    }
    __syncthreads();

    float local_sums[CHANNELS_PER_BLOCK] = {0.0f};
    
    // Main computation with warp-level reduction
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            const int h_in = h_out * stride + kh * dilation_h - pad_h;
            const bool valid_h = (h_in >= 0 && h_in < input_height);
            
            for (int kw = 0; kw < kernel_w; kw++) {
                const int w_in = w_out * stride + kw * dilation_w - pad_w;
                const bool valid_w = (w_in >= 0 && w_in < input_width);
                
                const float x_val = (valid_h && valid_w) ? 
                    __ldg(&x[b * in_channels * input_height * input_width +
                           ic * input_height * input_width +
                           h_in * input_width + w_in]) : 0.0f;

                #pragma unroll
                for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                    const int weight_idx = i * in_channels * kernel_h * kernel_w +
                                         ic * kernel_h * kernel_w +
                                         kh * kernel_w + kw;
                    local_sums[i] += x_val * shared_weight[weight_idx];
                }
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        float warp_sum = warp_reduce_sum(local_sums[i]);
        
        // First thread in each warp writes to shared memory
        if (lane_id == 0) {
            shared_partial_sums[warp_id * CHANNELS_PER_BLOCK + i] = warp_sum;
        }
    }
    __syncthreads();

    // Final reduction across warps (done by first warp)
    if (warp_id == 0 && lane_id < CHANNELS_PER_BLOCK) {
        float final_sum = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            final_sum += shared_partial_sums[w * CHANNELS_PER_BLOCK + lane_id];
        }
        
        const int global_oc = oc_start + lane_id;
        if (global_oc < out_channels) {
            if (bias != nullptr) {
                final_sum += bias[global_oc];
            }
            
            const int out_idx = b * out_channels * height_out * width_out +
                               global_oc * height_out * width_out +
                               h_out * width_out + w_out;
            output[out_idx] = final_sum;
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

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks((width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                batch_size * groups_per_batch);

    // Shared memory size includes space for weights and partial sums
    size_t shared_mem_size = (CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w +
                             ((BLOCK_SIZE_X * BLOCK_SIZE_Y) / WARP_SIZE) * CHANNELS_PER_BLOCK) * sizeof(float);

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