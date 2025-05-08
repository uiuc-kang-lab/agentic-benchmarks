#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized for H100 with 144 SMs
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define TILE_SIZE 4
#define SHARED_MEM_SIZE (BLOCK_DIM_X + 2) * (BLOCK_DIM_Y + 2) * TILE_SIZE

template <typename scalar_t>
__global__ void conv3d_balanced_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int groups) {

    __shared__ scalar_t shared_input[SHARED_MEM_SIZE];
    __shared__ scalar_t shared_weight[TILE_SIZE][TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output position
    const int out_x = bx * BLOCK_DIM_X + tx;
    const int out_y = by * BLOCK_DIM_Y + ty;
    const int out_c = bz % out_channels;
    const int batch = bz / out_channels;

    if (batch >= batch_size) return;

    const int group = out_c / (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;

    // Initialize accumulator for each thread
    scalar_t sum[TILE_SIZE][TILE_SIZE] = {0};

    // Process input in tiles
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        const int in_c = group * in_channels_per_group + ic;

        // Load weight tile into shared memory
        if (tx < kernel_w && ty < kernel_h) {
            for (int kd = 0; kd < kernel_d; kd++) {
                shared_weight[kd][ty][tx] = weight[
                    ((out_c * in_channels_per_group + ic) * kernel_d + kd) * 
                    kernel_h * kernel_w + ty * kernel_w + tx
                ];
            }
        }
        __syncthreads();

        // Process each depth level
        for (int d = 0; d < TILE_SIZE; d++) {
            const int out_d = blockIdx.z * TILE_SIZE + d;
            if (out_d >= out_depth) continue;

            // Load input tile into shared memory with padding
            for (int i = ty; i < BLOCK_DIM_Y + 2; i += BLOCK_DIM_Y) {
                for (int j = tx; j < BLOCK_DIM_X + 2; j += BLOCK_DIM_X) {
                    const int in_y = by * BLOCK_DIM_Y * stride + i - padding;
                    const int in_x = bx * BLOCK_DIM_X * stride + j - padding;
                    const int in_d = out_d * stride - padding;

                    if (in_y >= 0 && in_y < in_height && 
                        in_x >= 0 && in_x < in_width && 
                        in_d >= 0 && in_d < in_depth) {
                        shared_input[i * (BLOCK_DIM_X + 2) + j] = input[
                            ((batch * in_channels + in_c) * in_depth + in_d) * 
                            in_height * in_width + in_y * in_width + in_x
                        ];
                    } else {
                        shared_input[i * (BLOCK_DIM_X + 2) + j] = 0;
                    }
                }
            }
            __syncthreads();

            // Compute convolution for this tile
            #pragma unroll
            for (int kd = 0; kd < kernel_d; kd++) {
                #pragma unroll
                for (int kh = 0; kh < kernel_h; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; kw++) {
                        const int in_y = ty * stride + kh;
                        const int in_x = tx * stride + kw;
                        sum[d][0] += shared_input[(in_y) * (BLOCK_DIM_X + 2) + in_x] * 
                                   shared_weight[kd][kh][kw];
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write output with bias
    #pragma unroll
    for (int d = 0; d < TILE_SIZE; d++) {
        const int out_d = blockIdx.z * TILE_SIZE + d;
        if (out_d >= out_depth) continue;

        if (out_x < out_width && out_y < out_height) {
            const int out_idx = ((batch * out_channels + out_c) * out_depth + out_d) * 
                               out_height * out_width + out_y * out_width + out_x;
            output[out_idx] = sum[d][0] + (bias ? bias[out_c] : 0);
        }
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    auto bias = bias_opt.value_or(at::Tensor());

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_depth = (in_depth + 2 * padding - kernel_d) / stride + 1;
    const int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Calculate grid dimensions for balanced workload
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks(
        (out_width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (out_height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y,
        ((batch_size * out_channels + TILE_SIZE - 1) / TILE_SIZE)
    );

    conv3d_balanced_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with balanced workload (CUDA)");
}