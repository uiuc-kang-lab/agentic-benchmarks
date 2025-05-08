#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable parameters for optimal coalescing
#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 4
#define ELEMENTS_PER_THREAD 4  // Each thread computes 4 contiguous outputs along x

// This kernel uses shared memory tiling and vectorized global memory stores to ensure memory coalescing.
// It loads an input patch into shared memory so that threads can reuse data, and then each thread computes
// a small vector of outputs. Global memory writes are performed using float4 stores when possible to ensure
// that threads within a warp write to consecutive aligned memory addresses.

__global__ void coalesced_depthwise_conv2d_kernel(
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
    // Determine batch and channel indices. Each block in z corresponds to a (batch, channel) pair.
    int b = blockIdx.z / out_channels;
    int c = blockIdx.z % out_channels;
    int g = c / channels_per_group;  // group index
    int m = c % channels_per_group;  // channel index within the group

    // Output tile dimensions
    const int tile_out_width = BLOCK_DIM_X * ELEMENTS_PER_THREAD; // horizontal extent of output tile
    const int tile_out_height = BLOCK_DIM_Y;                        // vertical extent of output tile

    // Compute top-left coordinates of the output tile for this block
    int tile_out_x_origin = blockIdx.x * tile_out_width;
    int tile_out_y_origin = blockIdx.y * tile_out_height;

    // Compute corresponding top-left input coordinates (accounting for stride and padding)
    int in_x_origin = tile_out_x_origin * stride_w - padding_w;
    int in_y_origin = tile_out_y_origin * stride_h - padding_h;

    // Determine dimensions for the shared memory tile
    // It must cover the receptive field for the entire output tile
    int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = tile_out_height * stride_h + (kernel_h - 1) * dilation_h;

    // Allocate shared memory dynamically
    extern __shared__ float shared_input[];
    int shared_tile_size = shared_tile_width * shared_tile_height;

    // Load the input patch into shared memory. Use a single loop to have threads load consecutive elements.
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    for (int idx = thread_id; idx < shared_tile_size; idx += num_threads) {
        int sh_y = idx / shared_tile_width;
        int sh_x = idx % shared_tile_width;
        int in_x = in_x_origin + sh_x;
        int in_y = in_y_origin + sh_y;
        float val = 0.f;
        if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h) {
            // For depthwise convolution, the input channel corresponds to group index g
            int input_idx = ((b * in_channels + g) * in_h + in_y) * in_w + in_x;
            val = input[input_idx];
        }
        shared_input[idx] = val;
    }
    __syncthreads();

    // Each thread computes ELEMENTS_PER_THREAD output values for one row of the output tile
    int out_y = tile_out_y_origin + threadIdx.y;
    if (out_y < out_h) {
        int base_out_x = tile_out_x_origin + threadIdx.x * ELEMENTS_PER_THREAD;
        float results[ELEMENTS_PER_THREAD] = {0.f, 0.f, 0.f, 0.f};

        // Loop over kernel window
        for (int kh = 0; kh < kernel_h; kh++) {
            int sh_y = threadIdx.y * stride_h + kh * dilation_h;
            for (int kw = 0; kw < kernel_w; kw++) {
                float w_val = weight[((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw];
                int base_sh_x = threadIdx.x * ELEMENTS_PER_THREAD * stride_w + kw * dilation_w;
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                    int sh_x = base_sh_x + i * stride_w;
                    results[i] += shared_input[sh_y * shared_tile_width + sh_x] * w_val;
                }
            }
        }

        // Compute global output index for the first element this thread writes
        int out_row_index = ((b * out_channels + c) * out_h + out_y) * out_w;
        int out_index = out_row_index + base_out_x;

        // To ensure coalesced writes, use vectorized store with float4 if the address is 16-byte aligned
        if ((base_out_x + ELEMENTS_PER_THREAD) <= out_w && ((out_index % 4) == 0)) {
            if (bias != nullptr) {
                float bias_val = bias[c];
                results[0] += bias_val;
                results[1] += bias_val;
                results[2] += bias_val;
                results[3] += bias_val;
            }
            reinterpret_cast<float4*>(output)[out_index / 4] = make_float4(results[0], results[1], results[2], results[3]);
        } else {
            // Fallback to scalar stores for boundary conditions or misalignment cases
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                int out_x = base_out_x + i;
                if (out_x < out_w) {
                    float res = results[i];
                    if (bias != nullptr)
                        res += bias[c];
                    output[out_row_index + out_x] = res;
                }
            }
        }
    }
}


// Host function that sets up grid/block dimensions and launches the kernel

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
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value())
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");

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
    if (bias.has_value())
        bias_ptr = bias.value().data_ptr<float>();

    // Set up block and grid dimensions
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    int tile_out_width = BLOCK_DIM_X * ELEMENTS_PER_THREAD;
    int grid_x = (out_w + tile_out_width - 1) / tile_out_width;
    int grid_y = (out_h + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
    int grid_z = batch_size * out_channels;
    dim3 blocks(grid_x, grid_y, grid_z);

    // Calculate required shared memory size
    int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = BLOCK_DIM_Y * stride_h + (kernel_h - 1) * dilation_h;
    size_t shared_mem_size = shared_tile_width * shared_tile_height * sizeof(float);

    coalesced_depthwise_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Coalesced Depthwise Conv2D forward (CUDA) with aligned global memory accesses");
}
