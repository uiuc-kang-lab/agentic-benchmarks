#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable parameters
#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

// Combined CUDA kernel with improved global memory access using __ldg

__global__ void optimized_depthwise_conv2d_kernel(
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
    int b = blockIdx.z / out_channels;
    int c = blockIdx.z % out_channels;
    int g = c / channels_per_group;
    int m = c % channels_per_group;

    const int tile_out_width = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int tile_out_height = BLOCK_SIZE;

    int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = tile_out_height * stride_h + (kernel_h - 1) * dilation_h;
    int base_in_x = blockIdx.x * tile_out_width * stride_w - padding_w;
    int base_in_y = blockIdx.y * tile_out_height * stride_h - padding_h;

    extern __shared__ float shared_input[];
    int shared_size = shared_tile_width * shared_tile_height;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int thread_id = tidy * blockDim.x + tidx;
    int total_threads = blockDim.x * blockDim.y;

    for (int idx = thread_id; idx < shared_size; idx += total_threads) {
        int sh_y = idx / shared_tile_width;
        int sh_x = idx % shared_tile_width;
        int in_y = base_in_y + sh_y;
        int in_x = base_in_x + sh_x;
        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
            shared_input[sh_y * shared_tile_width + sh_x] = __ldg(
                &input[((b * in_channels + g) * in_h + in_y) * in_w + in_x]
            );
        } else {
            shared_input[sh_y * shared_tile_width + sh_x] = 0.0f;
        }
    }
    __syncthreads();

    int out_y = blockIdx.y * tile_out_height + tidy;
    if (out_y < out_h) {
        int base_out_x = blockIdx.x * tile_out_width + tidx * ELEMENTS_PER_THREAD;
        float results[ELEMENTS_PER_THREAD] = {0.f, 0.f, 0.f, 0.f};

        for (int kh = 0; kh < kernel_h; kh++) {
            int sh_y = tidy * stride_h + kh * dilation_h;
            for (int kw = 0; kw < kernel_w; kw++) {
                float w_val = __ldg(&weight[((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw]);
                int base_sh_x = tidx * ELEMENTS_PER_THREAD * stride_w + kw * dilation_w;
                #pragma unroll
                for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
                    int sh_x = base_sh_x + j * stride_w;
                    float in_val = shared_input[sh_y * shared_tile_width + sh_x];
                    results[j] += in_val * w_val;
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int out_x = base_out_x + j;
            if (out_x < out_w) {
                float res = results[j];
                if (bias != nullptr) {
                    res += __ldg(&bias[c]);
                }
                int out_idx = ((b * out_channels + c) * out_h + out_y) * out_w + out_x;
                output[out_idx] = res;
            }
        }
    }
}

// Host function with kernel launch setup

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

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_w + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD),
        (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );

    int tile_out_width = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int tile_out_height = BLOCK_SIZE;
    int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = tile_out_height * stride_h + (kernel_h - 1) * dilation_h;
    int shared_mem_size = shared_tile_width * shared_tile_height * sizeof(float);

    optimized_depthwise_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Optimized Depthwise Conv2D forward (CUDA) with global memory __ldg");
}