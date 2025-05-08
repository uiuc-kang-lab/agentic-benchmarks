#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 11
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

    __shared__ float shared_input[CHANNELS_PER_BLOCK][(TILE_SIZE + MAX_KERNEL_SIZE - 1)][(TILE_SIZE + MAX_KERNEL_SIZE - 1)];
    __shared__ float shared_weight[CHANNELS_PER_BLOCK][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int h_out_start = by * TILE_SIZE;
    int w_out_start = bx * TILE_SIZE;
    int b = bz / ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    int oc_start = (bz % ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)) * CHANNELS_PER_BLOCK;

    float sums[CHANNELS_PER_BLOCK] = {0.0f};

    // Initialize sums with bias if present
    #pragma unroll
    for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
        sums[oc_offset] = bias ? bias[oc_start + oc_offset] : 0.0f;
    }

    // Compute input tile size including padding for convolution
    int tile_h = TILE_SIZE + (kernel_h - 1) * dilation_h;
    int tile_w = TILE_SIZE + (kernel_w - 1) * dilation_w;

    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory
        for (int i = ty; i < tile_h; i += BLOCK_SIZE) {
            for (int j = tx; j < tile_w; j += BLOCK_SIZE) {
                int h_in = h_out_start * stride + i - pad_h;
                int w_in = w_out_start * stride + j - pad_w;
                
                float val = 0.0f;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    val = x[b * in_channels * input_height * input_width +
                          ic * input_height * input_width +
                          h_in * input_width + w_in];
                }
                shared_input[0][i][j] = val;
            }
        }

        // Load weights into shared memory
        for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
            for (int i = ty; i < kernel_h; i += BLOCK_SIZE) {
                for (int j = tx; j < kernel_w; j += BLOCK_SIZE) {
                    if (i < kernel_h && j < kernel_w) {
                        shared_weight[oc_offset][i][j] = weight[(oc_start + oc_offset) * in_channels * kernel_h * kernel_w +
                                                              ic * kernel_h * kernel_w +
                                                              i * kernel_w + j];
                    }
                }
            }
        }

        __syncthreads();

        // Compute convolution for this input channel
        int h_out = h_out_start + ty;
        int w_out = w_out_start + tx;

        if (h_out < height_out && w_out < width_out) {
            #pragma unroll
            for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
                float sum = 0.0f;
                #pragma unroll
                for (int kh = 0; kh < kernel_h; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int h_in_local = ty * stride + kh * dilation_h;
                        int w_in_local = tx * stride + kw * dilation_w;
                        sum += shared_input[0][h_in_local][w_in_local] * shared_weight[oc_offset][kh][kw];
                    }
                }
                sums[oc_offset] += sum;
            }
        }

        __syncthreads();
    }

    // Write results
    int h_out = h_out_start + ty;
    int w_out = w_out_start + tx;
    if (h_out < height_out && w_out < width_out) {
        #pragma unroll
        for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
            int out_idx = b * out_channels * height_out * width_out +
                         (oc_start + oc_offset) * height_out * width_out +
                         h_out * width_out + w_out;
            output[out_idx] = sums[oc_offset];
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

    TORCH_CHECK(kernel_h <= MAX_KERNEL_SIZE && kernel_w <= MAX_KERNEL_SIZE, 
               "Kernel size must be less than or equal to MAX_KERNEL_SIZE");

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (width_out + TILE_SIZE - 1) / TILE_SIZE,
        (height_out + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    conv2d_kernel<<<blocks, threads>>>(
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