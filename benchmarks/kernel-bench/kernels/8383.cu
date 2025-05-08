#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile size for input channels to load weight into shared memory
#define TILE_IN_CHANNELS 8

// Optimized kernel using 3D grid indexing and shared memory tiling for weight re-use
// Assumes dilation == 1 and groups == 1
__global__ void conv_transpose2d_kernel_tiled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int output_height,
    int output_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {

    // Compute spatial indices (output X and Y)
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    // Map grid.z to (batch, out_channel)
    int linear_idx = blockIdx.z;  // blockDim.z is 1
    int batch = linear_idx / out_channels;
    int out_ch = linear_idx % out_channels;

    if (out_x >= output_width || out_y >= output_height || batch >= batch_size)
        return;

    float sum = 0.0f;

    // Declare dynamic shared memory for a tile of weight data
    extern __shared__ float shared_weight[];  // size: TILE_IN_CHANNELS * kernel_height * kernel_width

    // Loop over input channels in tiles
    for (int in_ch_tile = 0; in_ch_tile < in_channels; in_ch_tile += TILE_IN_CHANNELS) {
        int tile_channels = ((in_ch_tile + TILE_IN_CHANNELS) <= in_channels) ? TILE_IN_CHANNELS : (in_channels - in_ch_tile);
        int weight_tile_size = tile_channels * kernel_height * kernel_width;

        // Cooperatively load a tile of weight for the current out_ch and tile of in_channels
        int threadsPerBlock = blockDim.x * blockDim.y;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = tid; i < weight_tile_size; i += threadsPerBlock) {
            int local_in_ch = i / (kernel_height * kernel_width);
            int rem = i % (kernel_height * kernel_width);
            int kh = rem / kernel_width;
            int kw = rem % kernel_width;
            int global_in_ch = in_ch_tile + local_in_ch;
            // Weight tensor assumed layout: (in_channels, out_channels, kernel_height, kernel_width)
            // Compute index for the weight corresponding to global_in_ch and fixed out_ch
            shared_weight[i] = weight[global_in_ch * (out_channels * kernel_height * kernel_width) +
                                        out_ch * (kernel_height * kernel_width) +
                                        kh * kernel_width + kw];
        }
        __syncthreads();

        // For each channel in the tile, accumulate the contributions
        for (int local_in_ch = 0; local_in_ch < tile_channels; local_in_ch++) {
            int global_in_ch = in_ch_tile + local_in_ch;
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = out_x + pad_w - kw;
                    int in_y = out_y + pad_h - kh;
                    // Check for proper alignment according to stride
                    if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                        in_x /= stride_w;
                        in_y /= stride_h;
                        if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                            float input_val = input[batch * in_channels * input_height * input_width +
                                                      global_in_ch * input_height * input_width +
                                                      in_y * input_width + in_x];
                            float weight_val = shared_weight[local_in_ch * (kernel_height * kernel_width) + kh * kernel_width + kw];
                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }
        __syncthreads(); // Ensure tile memory is freed before next iteration
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[out_ch];
    }

    // Write the computed output value
    output[batch * out_channels * output_height * output_width +
           out_ch * output_height * output_width +
           out_y * output_width + out_x] = sum;
}


// Host function that prepares and launches the tiled kernel
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    // For simplicity, this implementation supports dilation == 1 and groups == 1
    TORCH_CHECK(dilation[0] == 1 && dilation[1] == 1, "Only dilation=1 is supported in tiled_conv_transpose kernel");
    TORCH_CHECK(groups == 1, "Only groups=1 is supported in tiled_conv_transpose kernel");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    // Configure threads and blocks: use 2D threads for spatial dimensions and map batch*out_channels to grid.z
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels // each thread in z covers one (batch, channel) pair
    );

    // Set up pointers for bias
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Calculate shared memory size: maximum tile is TILE_IN_CHANNELS channels
    size_t shared_mem_size = TILE_IN_CHANNELS * kernel_height * kernel_width * sizeof(float);

    conv_transpose2d_kernel_tiled<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Tiled and optimized ConvTranspose2D forward (CUDA)");
}
