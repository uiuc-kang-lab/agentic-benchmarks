#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256

// Optimized kernel combining efficient memory access and atomic operations
__global__ void optimized_conv_transpose2d_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
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
    int pad_w,
    int total_sum_elems,  // = in_channels * kernel_height * kernel_width
    int blocks_per_out)   // number of blocks to partition summation for each output pixel
{
    int global_blk_idx = blockIdx.x;
    int out_pixel_idx = global_blk_idx / blocks_per_out;
    int block_in_pixel = global_blk_idx % blocks_per_out;

    int tmp = out_pixel_idx;
    int out_x = tmp % output_width; tmp /= output_width;
    int out_y = tmp % output_height; tmp /= output_height;
    int out_ch = tmp % out_channels; tmp /= out_channels;
    int batch = tmp;

    float local_sum = 0.0f;
    for (int i = block_in_pixel + threadIdx.x * blocks_per_out; i < total_sum_elems; i += blocks_per_out * blockDim.x) {
        int ic = i / (kernel_height * kernel_width);
        int rem = i % (kernel_height * kernel_width);
        int kh = rem / kernel_width;
        int kw = rem % kernel_width;

        int in_x = out_x + pad_w - kw;
        int in_y = out_y + pad_h - kh;
        if ((in_x % stride_w) == 0 && (in_y % stride_h) == 0) {
            in_x /= stride_w;
            in_y /= stride_h;
            if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                float inp_val = input[batch * in_channels * input_height * input_width +
                                        ic * input_height * input_width +
                                        in_y * input_width + in_x];
                float w = weight[ic * out_channels * kernel_height * kernel_width +
                                 out_ch * kernel_height * kernel_width +
                                 kh * kernel_width + kw];
                local_sum += inp_val * w;
            }
        }
    }

    __shared__ float smem[THREADS];
    int tid = threadIdx.x;
    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float block_sum = smem[0];
        if (block_in_pixel == 0 && bias != nullptr) {
            block_sum += bias[out_ch];
        }
        int out_idx = batch * out_channels * output_height * output_width +
                      out_ch * output_height * output_width +
                      out_y * output_width + out_x;
        atomicAdd(&output[out_idx], block_sum);
    }
}

// Host function to prepare and launch the kernel
torch::Tensor optimized_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width  = (input_width - 1)  * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    int total_sum_elems = in_channels * kernel_height * kernel_width;
    int blocks_per_out = (total_sum_elems + THREADS - 1) / THREADS;

    int num_output_pixels = batch_size * out_channels * output_height * output_width;
    int total_blocks = num_output_pixels * blocks_per_out;

    dim3 blocks(total_blocks);
    dim3 threads(THREADS);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    optimized_conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
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
        padding[1],
        total_sum_elems,
        blocks_per_out
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_conv_transpose2d, "Optimized ConvTranspose2D forward (CUDA)");
}
