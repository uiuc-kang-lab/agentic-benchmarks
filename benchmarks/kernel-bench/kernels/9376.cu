#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define CHANNELS_PER_BLOCK 4
#define WARP_SIZE 32

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

    __shared__ float s_weight[CHANNELS_PER_BLOCK][TILE_SIZE * TILE_SIZE];
    __shared__ float s_input[TILE_SIZE][TILE_SIZE + 2];  // +2 for padding

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate batch and output channel indices
    const int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    const int b = bz / groups_per_batch;
    const int oc_group = bz % groups_per_batch;
    const int oc_start = oc_group * CHANNELS_PER_BLOCK;

    // Calculate output coordinates
    const int h_out_start = by * TILE_SIZE + ty;
    const int w_out_start = bx * TILE_SIZE + tx;

    // Early exit if outside output bounds
    if (h_out_start >= height_out || w_out_start >= width_out || b >= batch_size) return;

    // Initialize accumulators
    float sums[CHANNELS_PER_BLOCK] = {0.0f};
    if (bias != nullptr) {
        #pragma unroll
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            const int oc = oc_start + i;
            if (oc < out_channels) {
                sums[i] = bias[oc];
            }
        }
    }

    // Load weights into shared memory - only once per block
    const int thread_idx = ty * TILE_SIZE + tx;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        for (int j = thread_idx; j < kernel_h * kernel_w * in_channels; j += total_threads) {
            const int oc = oc_start + i;
            if (oc < out_channels) {
                s_weight[i][j] = weight[oc * in_channels * kernel_h * kernel_w + j];
            }
        }
    }
    __syncthreads();  // Ensure weights are loaded

    // Main computation loop
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            const int h_in = h_out_start * stride + kh * dilation_h - pad_h;
            
            if (h_in >= 0 && h_in < input_height) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int w_in = w_out_start * stride + kw * dilation_w - pad_w;
                    
                    if (w_in >= 0 && w_in < input_width) {
                        const float x_val = __ldg(&x[((b * in_channels + ic) * input_height + h_in) * input_width + w_in]);
                        
                        #pragma unroll
                        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                            if (oc_start + i < out_channels) {
                                const int weight_idx = (ic * kernel_h + kh) * kernel_w + kw;
                                sums[i] += x_val * s_weight[i][weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        const int oc = oc_start + i;
        if (oc < out_channels) {
            const int out_idx = ((b * out_channels + oc) * height_out + h_out_start) * width_out + w_out_start;
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks((width_out + TILE_SIZE - 1) / TILE_SIZE,
                (height_out + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * groups_per_batch);

    const size_t shared_mem_size = 
        (CHANNELS_PER_BLOCK * TILE_SIZE * TILE_SIZE + 
         TILE_SIZE * (TILE_SIZE + 2)) * sizeof(float);

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