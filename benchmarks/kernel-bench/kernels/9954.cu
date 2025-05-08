#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_WIDTH 32
#define TILE_HEIGHT 4

__global__ void depthwise_conv2d_kernel(
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
    // Each warp processes consecutive elements in the width dimension
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Block handles a tile of the output
    const int tile_start_x = blockIdx.x * TILE_WIDTH;
    const int tile_start_y = blockIdx.y * TILE_HEIGHT;
    
    // Each thread processes multiple elements for better instruction-level parallelism
    const int c_out = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;
    
    const int g = c_out / channels_per_group;
    const int m = c_out % channels_per_group;

    // Pre-compute weight indices for the current output channel
    const float* weight_c = &weight[(g * channels_per_group + m) * kernel_h * kernel_w];

    // Process multiple rows within the tile
    #pragma unroll
    for (int ty = 0; ty < TILE_HEIGHT; ty++) {
        const int y = tile_start_y + ty;
        if (y >= out_h) continue;

        // Process elements within a row in a coalesced manner
        #pragma unroll 4
        for (int tx = lane; tx < TILE_WIDTH; tx += WARP_SIZE) {
            const int x = tile_start_x + tx;
            if (x >= out_w) continue;

            float sum = 0.0f;
            
            // Compute convolution for this output position
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                const int h_in = y * stride_h - padding_h + kh * dilation_h;
                if (h_in >= 0 && h_in < in_h) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; kw++) {
                        const int w_in = x * stride_w - padding_w + kw * dilation_w;
                        if (w_in >= 0 && w_in < in_w) {
                            // Coalesced read from input
                            const int in_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                            const int w_idx = kh * kernel_w + kw;
                            sum += input[in_idx] * weight_c[w_idx];
                        }
                    }
                }
            }

            if (bias != nullptr) {
                sum += bias[c_out];
            }

            // Coalesced write to output
            const int out_idx = ((b * out_channels + c_out) * out_h + y) * out_w + x;
            output[out_idx] = sum;
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

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Configure grid and block dimensions for coalesced memory access
    dim3 threads(WARP_SIZE * (TILE_HEIGHT > 1 ? 2 : 1)); // Multiple warps per block
    dim3 blocks(
        (out_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * out_channels
    );

    depthwise_conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise Conv2D forward with coalesced memory access (CUDA)");
}