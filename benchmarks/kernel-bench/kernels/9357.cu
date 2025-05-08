#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define CHANNELS_PER_BLOCK 8

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

    // Shared memory for input tile and weights
    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;
    
    // Calculate indices
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Block handles a 2D tile of output and multiple output channels
    int out_x_base = blockIdx.x * BLOCK_SIZE_X * 4; // Each thread processes 4 elements horizontally
    int out_y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int b = blockIdx.z / ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    int oc_group = blockIdx.z % ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    int oc_start = oc_group * CHANNELS_PER_BLOCK;

    // Pre-compute input positions for the entire tile
    int in_y = out_y * stride - pad_h;

    // Load weights into shared memory (coalesced access)
    int weights_per_thread = (CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w + blockDim.x * blockDim.y - 1) 
                            / (blockDim.x * blockDim.y);
    for (int i = 0; i < weights_per_thread; i++) {
        int idx = tid + i * blockDim.x * blockDim.y;
        if (idx < CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w) {
            int oc_offset = idx / (in_channels * kernel_h * kernel_w);
            int remainder = idx % (in_channels * kernel_h * kernel_w);
            if (oc_start + oc_offset < out_channels) {
                shared_weight[idx] = weight[(oc_start + oc_offset) * in_channels * kernel_h * kernel_w + remainder];
            }
        }
    }
    __syncthreads();

    // Process 4 horizontal output positions per thread for better memory coalescing
    float sums[CHANNELS_PER_BLOCK][4] = {0};
    
    // Initialize with bias if present
    #pragma unroll
    for (int oc = 0; oc < CHANNELS_PER_BLOCK; oc++) {
        if (oc_start + oc < out_channels) {
            float bias_val = bias ? bias[oc_start + oc] : 0.0f;
            #pragma unroll
            for (int x_offset = 0; x_offset < 4; x_offset++) {
                sums[oc][x_offset] = bias_val;
            }
        }
    }

    if (out_y < height_out) {
        // Compute convolution
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                int in_y_pos = in_y + kh * dilation_h;
                if (in_y_pos >= 0 && in_y_pos < input_height) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        // Process 4 horizontal output positions
                        #pragma unroll
                        for (int x_offset = 0; x_offset < 4; x_offset++) {
                            int out_x = out_x_base + threadIdx.x * 4 + x_offset;
                            if (out_x < width_out) {
                                int in_x = out_x * stride + kw * dilation_w - pad_w;
                                if (in_x >= 0 && in_x < input_width) {
                                    float in_val = __ldg(&x[b * in_channels * input_height * input_width +
                                                         ic * input_height * input_width +
                                                         in_y_pos * input_width + in_x]);
                                    
                                    #pragma unroll
                                    for (int oc = 0; oc < CHANNELS_PER_BLOCK; oc++) {
                                        if (oc_start + oc < out_channels) {
                                            float w_val = shared_weight[oc * in_channels * kernel_h * kernel_w +
                                                                      ic * kernel_h * kernel_w +
                                                                      kh * kernel_w + kw];
                                            sums[oc][x_offset] += in_val * w_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Write output (coalesced access)
        #pragma unroll
        for (int x_offset = 0; x_offset < 4; x_offset++) {
            int out_x = out_x_base + threadIdx.x * 4 + x_offset;
            if (out_x < width_out) {
                #pragma unroll
                for (int oc = 0; oc < CHANNELS_PER_BLOCK; oc++) {
                    if (oc_start + oc < out_channels) {
                        output[b * out_channels * height_out * width_out +
                              (oc_start + oc) * height_out * width_out +
                              out_y * width_out + out_x] = sums[oc][x_offset];
                    }
                }
            }
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

    // Configure grid and blocks for coalesced memory access
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (width_out + BLOCK_SIZE_X * 4 - 1) / (BLOCK_SIZE_X * 4),
        (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    // Shared memory size for weights
    size_t shared_mem_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w * sizeof(float);

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