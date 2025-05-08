#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block size for spatial tile and number of output channels computed per block
#define BLOCK_SIZE 16
#define CHANNELS_PER_BLOCK 4

// The kernel leverages shared memory to cache both the input tile and the weight tile for a group of output channels.
// This reduces global memory latency by reusing data across threads in the block and minimizes the number of global memory accesses.

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

    // Determine the output spatial coordinate for this thread
    int out_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int out_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Decode blockIdx.z to obtain the batch index and output channel group
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = blockIdx.z / groups_per_batch;
    int oc_group = blockIdx.z % groups_per_batch;
    int oc_start = oc_group * CHANNELS_PER_BLOCK;

    // Return if the output coordinate is out of bounds
    if (out_y >= height_out || out_x >= width_out || b >= batch_size) return;

    // Compute the dimensions of the input tile needed for the output tile computed by this block
    int tile_h = (BLOCK_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int tile_w = (BLOCK_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;

    // Compute the starting global input coordinates for the shared input tile
    int in_y_start = blockIdx.y * BLOCK_SIZE * stride - pad_h;
    int in_x_start = blockIdx.x * BLOCK_SIZE * stride - pad_w;

    // Allocate shared memory (dynamically allocated) and partition it:
    // First portion stores the input tile for all input channels: size = in_channels * tile_h * tile_w
    // Second portion stores the weight tile for the current group of output channels: size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w
    extern __shared__ float smem[];
    int smem_input_size = in_channels * tile_h * tile_w;
    float* smem_input = smem;
    float* smem_weight = smem + smem_input_size;

    // Use all threads in the block (BLOCK_SIZE x BLOCK_SIZE) to cooperatively load shared memory.
    int block_threads = BLOCK_SIZE * BLOCK_SIZE;
    int tid = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    // Load the input tile from global memory into shared memory
    int total_input_elems = smem_input_size;
    for (int i = tid; i < total_input_elems; i += block_threads) {
        int ic = i / (tile_h * tile_w);
        int rem = i % (tile_h * tile_w);
        int sh = rem / tile_w;
        int sw = rem % tile_w;
        int global_y = in_y_start + sh;
        int global_x = in_x_start + sw;
        float val = 0.0f;
        if (global_y >= 0 && global_y < input_height && global_x >= 0 && global_x < input_width) {
            int idx = b * in_channels * input_height * input_width +
                      ic * input_height * input_width +
                      global_y * input_width + global_x;
            val = x[idx];
        }
        smem_input[i] = val;
    }

    // Load the weight tile for the current output channel group into shared memory
    int weight_tile_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    int total_weight_elems = weight_tile_size;
    for (int i = tid; i < total_weight_elems; i += block_threads) {
        int oc_offset = i / (in_channels * kernel_h * kernel_w);
        int rem = i % (in_channels * kernel_h * kernel_w);
        int ic = rem / (kernel_h * kernel_w);
        int rem2 = rem % (kernel_h * kernel_w);
        int kh = rem2 / kernel_w;
        int kw = rem2 % kernel_w;
        int oc = oc_start + oc_offset;
        float w_val = 0.0f;
        if (oc < out_channels) {
            int w_idx = oc * in_channels * kernel_h * kernel_w +
                        ic * kernel_h * kernel_w +
                        kh * kernel_w + kw;
            w_val = weight[w_idx];
        }
        smem_weight[i] = w_val;
    }

    __syncthreads();

    // Each thread computes its output for each output channel in the group
    float accum[CHANNELS_PER_BLOCK];
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            accum[i] = (bias != nullptr) ? bias[global_oc] : 0.0f;
        } else {
            accum[i] = 0.0f;
        }
    }

    // Compute the relative position in the shared input tile for the output element computed by this thread
    // Global input coordinate for the top-left of the output's receptive field is: out_y * stride + kh*dilation_h
    // The corresponding index in shared memory is offset by the tile's starting position (in_y_start)
    int rel_y = out_y * stride - in_y_start;
    int rel_x = out_x * stride - in_x_start;

    // Loop over all input channels and kernel elements, accumulating contributions via shared memory
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_sh = rel_y + kh * dilation_h;
                int in_sw = rel_x + kw * dilation_w;
                float input_val = smem_input[ ic * (tile_h * tile_w) + in_sh * tile_w + in_sw ];
                #pragma unroll
                for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                    int weight_idx = i * (in_channels * kernel_h * kernel_w) +
                                     ic * (kernel_h * kernel_w) +
                                     kh * kernel_w + kw;
                    accum[i] += input_val * smem_weight[weight_idx];
                }
            }
        }
    }

    // Write the computed outputs to global memory
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int oc = oc_start + i;
        if (oc < out_channels) {
            int out_idx = b * out_channels * height_out * width_out +
                          oc * height_out * width_out +
                          out_y * width_out + out_x;
            output[out_idx] = accum[i];
        }
    }
}


// PyBind11 binding and forward function

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

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks((width_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size * groups_per_batch);

    // Calculate shared memory size: space for the input tile + weight tile
    int tile_h = (BLOCK_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int tile_w = (BLOCK_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    size_t smem_input_size = in_channels * tile_h * tile_w * sizeof(float);
    size_t smem_weight_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t shared_mem_size = smem_input_size + smem_weight_size;

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
