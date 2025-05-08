#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define CHANNELS_PER_BLOCK 4

// Optimized kernel combining shared memory usage and loop unrolling
__global__ void optimized_conv2d_kernel(
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

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // Decompose blockIdx.z into batch index and output channel group
    int groups = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = bz / groups;
    int oc_group = bz % groups;
    int oc_start = oc_group * CHANNELS_PER_BLOCK;

    int h_out_idx = by * TILE_SIZE + ty;
    int w_out_idx = bx * TILE_SIZE + tx;

    if (h_out_idx >= height_out || w_out_idx >= width_out || b >= batch_size) return;

    // Shared memory for input tile
    extern __shared__ float shared_input[];
    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width  = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;

    // Cooperative loading into shared memory
    int total_shared_elems = in_channels * shared_height * shared_width;
    int thread_id = ty * blockDim.x + tx;
    int block_threads = blockDim.x * blockDim.y;

    for (int i = thread_id; i < total_shared_elems; i += block_threads) {
        int ic = i / (shared_height * shared_width);
        int rem = i % (shared_height * shared_width);
        int sh = rem / shared_width;
        int sw = rem % shared_width;

        int global_row = by * TILE_SIZE * stride - pad_h + sh;
        int global_col = bx * TILE_SIZE * stride - pad_w + sw;
        float val = 0.0f;
        if (global_row >= 0 && global_row < input_height && global_col >= 0 && global_col < input_width) {
            int x_index = b * in_channels * input_height * input_width
                        + ic * input_height * input_width
                        + global_row * input_width + global_col;
            val = x[x_index];
        }
        shared_input[i] = val;
    }

    __syncthreads();

    // Initialize accumulation with bias
    float sums[CHANNELS_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        if (oc_start + i < out_channels) {
            sums[i] = bias ? bias[oc_start + i] : 0.0f;
        }
    }

    // Loop over input channels and kernel dimensions
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int sh = ty * stride + kh * dilation_h;
            #pragma unroll
            for (int kw = 0; kw < kernel_w; ++kw) {
                int sw = tx * stride + kw * dilation_w;
                float input_val = shared_input[ic * (shared_height * shared_width) + sh * shared_width + sw];

                #pragma unroll
                for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                    if (oc_start + i < out_channels) {
                        float w_val = weight[(oc_start + i) * in_channels * kernel_h * kernel_w +
                                             ic * kernel_h * kernel_w +
                                             kh * kernel_w + kw];
                        sums[i] += input_val * w_val;
                    }
                }
            }
        }
    }

    // Write the computed output for each output channel in the group
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        if (oc_start + i < out_channels) {
            int out_idx = b * out_channels * height_out * width_out +
                          (oc_start + i) * height_out * width_out +
                          h_out_idx * width_out + w_out_idx;
            output[out_idx] = sums[i];
        }
    }
}

// PyBind11 binding

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
    int groups = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks((width_out + TILE_SIZE - 1) / TILE_SIZE,
                (height_out + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * groups);

    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width  = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    size_t shared_mem_size = in_channels * shared_height * shared_width * sizeof(float);

    optimized_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
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
