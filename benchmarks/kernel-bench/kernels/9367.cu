/*
Combining shared memory weight loading (Kernel 1) with warp-level reduction (Kernel 2).
Each block (a warp of 32 threads) computes one output pixel for a group of output channels.
Weights for the output channel group are loaded once into shared memory and then used by all threads,
allowing an efficient warp-level reduction over the convolution reduction dimension.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Process this many output channels per block
#define CHANNELS_PER_BLOCK 4
#define WARP_SIZE 32

// New Kernel: Each warp loads its block of weights to shared memory and then computes the convolution
// using warp-level reduction. This combines the benefits of reduced global memory access 
// (via shared memory) and efficient inter-thread reduction.

__global__ void conv2d_sharedwarp_kernel(
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

    // Each block corresponds to one output pixel (h_out, w_out) for a particular batch and channel group.
    // Block grid: x -> width_out, y -> height_out, z -> batch index and channel group.

    // Thread lane within a warp
    int lane = threadIdx.x;  // expecting blockDim.x == WARP_SIZE

    // Determine output spatial coordinates
    int w_out = blockIdx.x;
    int h_out = blockIdx.y;

    if (h_out >= height_out || w_out >= width_out) return;

    // Decode batch and output channel group from blockIdx.z
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = blockIdx.z / groups_per_batch;
    int g = blockIdx.z % groups_per_batch;
    int oc_start = g * CHANNELS_PER_BLOCK;

    // Allocate shared memory to hold the weight slice for this output channel group
    // The shared memory size: CHANNELS_PER_BLOCK * (in_channels * kernel_h * kernel_w) floats
    extern __shared__ float shared_weight[];
    int weight_block_elems = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;

    // Each thread in the warp cooperatively loads weights from global to shared memory
    for (int idx = lane; idx < weight_block_elems; idx += WARP_SIZE) {
        int sub_channel = idx / (in_channels * kernel_h * kernel_w);
        int rem = idx % (in_channels * kernel_h * kernel_w);
        int global_oc = oc_start + sub_channel;
        if (global_oc < out_channels) {
            // Global weight index calculation:
            // weight is laid out as [out_channels, in_channels, kernel_h, kernel_w]
            shared_weight[idx] = weight[global_oc * (in_channels * kernel_h * kernel_w) + rem];
        } else {
            shared_weight[idx] = 0.0f;
        }
    }
    __syncwarp(); // Synchronize within the warp

    // Each thread holds partial sums for each of the CHANNELS_PER_BLOCK output channels
    float partial[CHANNELS_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Total number of multiplications per output pixel (over in_channels and kernel area)
    int reduction_length = in_channels * kernel_h * kernel_w;

    // Each thread processes a strided subset of the reduction dimension
    for (int idx = lane; idx < reduction_length; idx += WARP_SIZE) {
        int ic = idx / (kernel_h * kernel_w);
        int rem = idx % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        // Compute the corresponding input coordinates
        int h_in = h_out * stride + kh * dilation_h - pad_h;
        int w_in = w_out * stride + kw * dilation_w - pad_w;
        float x_val = 0.0f;
        if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
            x_val = __ldg(&x[b * in_channels * input_height * input_width +
                             ic * input_height * input_width +
                             h_in * input_width + w_in]);
        }

        // For each output channel in this group use the preloaded weight
        // The weight is stored in shared memory with layout:
        // [channel in group][ic][kh][kw] contiguous per channel
        #pragma unroll
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            int shared_idx = i * (in_channels * kernel_h * kernel_w) + idx;
            partial[i] += x_val * shared_weight[shared_idx];
        }
    }

    // Perform warp-level reduction to sum partial results across threads
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            partial[i] += __shfl_down_sync(mask, partial[i], offset);
        }
    }

    // Lane 0 writes the final output for the computed output pixel
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
            int global_oc = oc_start + i;
            if (global_oc < out_channels) {
                int out_idx = b * out_channels * height_out * width_out +
                              global_oc * height_out * width_out +
                              h_out * width_out + w_out;
                float bias_val = (bias != nullptr) ? bias[global_oc] : 0.f;
                output[out_idx] = bias_val + partial[i];
            }
        }
    }
}

// Forward function callable from PyTorch

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // bias is optional
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

    // Grid dimensions: x -> width_out, y -> height_out,
    // z dimension encodes both batch and channel group information.
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks(width_out, height_out, batch_size * groups_per_batch);
    // Each block is one warp
    dim3 threads(WARP_SIZE, 1, 1);

    // Shared memory: weight block for one output channel group
    size_t shared_mem_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w * sizeof(float);

    conv2d_sharedwarp_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Conv2D forward with shared memory and warp-level reduction (CUDA)");
}
