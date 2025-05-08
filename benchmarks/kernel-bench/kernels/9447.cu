#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Experimenting with block sizes: using 32x16 threads per block (512 threads per block).
// This kernel maps each block to a specific (w_out, h_out) spatial location and a group of output channels
// for a specific batch sample. Batch and channel group are mapped into the grid z-dimension.

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define CHANNELS_PER_BLOCK 4

__global__ void conv2d_kernel_experiment(
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

    // Compute output spatial coordinates
    int w_out = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int h_out = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

    // Shared memory for input and weights
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem;
    float* shared_weight = shared_mem + BLOCK_SIZE_X * BLOCK_SIZE_Y;

    // Load input and weights into shared memory
    int h_in_start = h_out * stride - pad_h;
    int w_in_start = w_out * stride - pad_w;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_in_start + kh * dilation_h;
            if (h_in >= 0 && h_in < input_height) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_in_start + kw * dilation_w;
                    if (w_in >= 0 && w_in < input_width) {
                        shared_x[threadIdx.y * BLOCK_SIZE_X + threadIdx.x] = x[b * in_channels * input_height * input_width +
                                                                             ic * input_height * input_width +
                                                                             h_in * input_width + w_in];
                    }
                }
            }
        }
    }

    for (int oc = oc_base; oc < oc_base + CHANNELS_PER_BLOCK && oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    shared_weight[threadIdx.y * BLOCK_SIZE_X + threadIdx.x] = weight[oc * in_channels * kernel_h * kernel_w +
                                                                                     ic * kernel_h * kernel_w +
                                                                                     kh * kernel_w + kw];
                }
            }
        }
    }

    __syncthreads();

    // Compute batch and output channel group indices from grid z-dimension
    int groupCount = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;  // total channel groups
    int b = blockIdx.z / groupCount; // batch index
    int oc_group = blockIdx.z % groupCount;  
    int oc_base = oc_group * CHANNELS_PER_BLOCK;

    if (b >= batch_size || h_out >= height_out || w_out >= width_out) return;

    // Initialize accumulators with bias if available
    float sums[CHANNELS_PER_BLOCK];
    #pragma unroll
    for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
        int oc = oc_base + k;
        sums[k] = (bias && oc < out_channels) ? bias[oc] : 0.0f;
    }

    // Iterate over input channels and kernel spatial dimensions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_out * stride + kh * dilation_h - pad_h;
            if (h_in < 0 || h_in >= input_height) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                if (w_in < 0 || w_in >= input_width) continue;
                float x_val = x[b * in_channels * input_height * input_width +
                                ic * input_height * input_width +
                                h_in * input_width + w_in];
                
                #pragma unroll
                for (int k = 0; k < CHANNELS_PER_BLOCK; ++k) {
                    int oc = oc_base + k;
                    if (oc >= out_channels) break;
                    float w_val = weight[ oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw];
                    sums[k] += x_val * w_val;
                }
            }
        }
    }

    // Write computed sums to output tensor
    #pragma unroll
    for (int k = 0; k < CHANNELS_PER_BLOCK; ++k) {
        int oc = oc_base + k;
        if (oc >= out_channels) break;
        int out_idx = b * out_channels * height_out * width_out +
                      oc * height_out * width_out +
                      h_out * width_out + w_out;
        output[out_idx] = sums[k];
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
    if (height_out == 0 || width_out == 0) return output;

    // Determine grid dimensions
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int groupCount = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;  // number of channel groups per batch
    dim3 blocks(
        (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * groupCount
    );

    conv2d_kernel_experiment<<<blocks, threads>>>(
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
