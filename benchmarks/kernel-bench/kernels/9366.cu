#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHANNELS_PER_BLOCK 4
#define WARP_SIZE 32

// Kernel: Combines shared memory usage with warp-efficient reduction
__global__ void conv2d_optimized_kernel(
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

    int lane = threadIdx.x;
    int w_out = blockIdx.x; 
    int h_out = blockIdx.y; 
    if (h_out >= height_out || w_out >= width_out) return;

    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = blockIdx.z / groups_per_batch; 
    int g = blockIdx.z % groups_per_batch;
    int oc_start = g * CHANNELS_PER_BLOCK;

    float partial[CHANNELS_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};
    int reduction_length = in_channels * kernel_h * kernel_w;

    extern __shared__ float shared_weight[];
    int threadId = threadIdx.x;
    for (int idx = threadId; idx < CHANNELS_PER_BLOCK * reduction_length; idx += WARP_SIZE) {
        int w_oc = idx / (reduction_length);
        int rem = idx % (reduction_length);
        int global_oc = oc_start + w_oc;
        if (global_oc < out_channels) {
            shared_weight[idx] = weight[global_oc * reduction_length + rem];
        }
    }
    __syncthreads();

    for (int idx = lane; idx < reduction_length; idx += WARP_SIZE) {
        int ic = idx / (kernel_h * kernel_w);
        int rem = idx % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        int h_in = h_out * stride + kh * dilation_h - pad_h;
        int w_in = w_out * stride + kw * dilation_w - pad_w;
        
        if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
            float x_val = __ldg(&x[b * in_channels * input_height * input_width +
                                   ic * input_height * input_width +
                                   h_in * input_width + w_in]);
            
            #pragma unroll
            for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                int weight_offset = i * reduction_length + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                partial[i] += x_val * shared_weight[weight_offset];
            }
        }
    }

    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            partial[i] += __shfl_down_sync(mask, partial[i], offset);
        }
    }

    if (lane == 0) {
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            int global_oc = oc_start + i;
            if (global_oc < out_channels) {
                int out_idx = b * out_channels * height_out * width_out +
                              global_oc * height_out * width_out +
                              h_out * width_out + w_out;
                float res = (bias != nullptr) ? bias[global_oc] : 0.0f;
                output[out_idx] = res + partial[i];
            }
        }
    }
}

// Forward function

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
    int width_out  = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks(width_out, height_out, batch_size * groups_per_batch);
    dim3 threads(WARP_SIZE, 1, 1);

    size_t shared_mem_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w * sizeof(float);

    conv2d_optimized_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Optimized Conv2D forward (CUDA)");
}
