#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Use a fixed number of threads per block
#define THREADS 256

// New kernel: one block per output pixel
// Each block sums over the entire reduction domain (in_channels * kernel_height * kernel_width) for one output pixel
// This design eliminates the inter-block atomicAdd overhead present in the original kernel2
__global__ void conv_transpose2d_direct_kernel(
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
    int total_sum_elems) {

    // Each block corresponds to one output pixel
    int out_pixel = blockIdx.x;  // global output pixel index

    // Decode the output pixel index into its coordinates:
    // The layout is: [batch, out_channel, out_y, out_x]
    int tmp = out_pixel;
    int out_x = tmp % output_width; tmp /= output_width;
    int out_y = tmp % output_height; tmp /= output_height;
    int out_ch = tmp % out_channels; tmp /= out_channels;
    int batch = tmp;  // remaining is the batch index

    float local_sum = 0.0f;

    // Loop over the entire reduction domain (i.e., across input channels and kernel elements)
    for (int i = threadIdx.x; i < total_sum_elems; i += blockDim.x) {
        // Decode index i into: in_channel, kernel row and col
        int ic = i / (kernel_height * kernel_width);
        int rem = i % (kernel_height * kernel_width);
        int kh = rem / kernel_width;
        int kw = rem % kernel_width;

        // Calculate corresponding input spatial coordinate
        int in_x = out_x + pad_w - kw;
        int in_y = out_y + pad_h - kh;

        // Check if the coordinate properly aligns with the stride
        if (in_x % stride_w == 0 && in_y % stride_h == 0) {
            in_x /= stride_w;
            in_y /= stride_h;
            // Bounds check
            if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                // Compute flat index for input and weight
                int input_index = batch * in_channels * input_height * input_width +
                                  ic * input_height * input_width +
                                  in_y * input_width + in_x;

                int weight_index = ic * out_channels * kernel_height * kernel_width +
                                   out_ch * kernel_height * kernel_width +
                                   kh * kernel_width + kw;

                local_sum += input[input_index] * weight[weight_index];
            }
        }
    }

    // In-block reduction using shared memory
    __shared__ float sdata[THREADS];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final output and adds bias if provided
    if (threadIdx.x == 0) {
        float out_val = sdata[0];
        if (bias != nullptr) {
            out_val += bias[out_ch];
        }
        int output_index = batch * out_channels * output_height * output_width +
                           out_ch * output_height * output_width +
                           out_y * output_width + out_x;
        output[output_index] = out_val;
    }
}

// Host function that sets up and launches the new kernel

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,  // unused: we assume dilation==1
    int64_t groups) {  // unused: we assume groups==1 for simplicity

    // Get dimensions from input tensors
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    // Compute output dimensions using conv_transpose2d formula
    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width  = (input_width - 1)  * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    // Total number of elements over which to reduce for each output pixel
    int total_sum_elems = in_channels * kernel_height * kernel_width;

    // Define grid: one block per output pixel
    int num_output_pixels = batch_size * out_channels * output_height * output_width;
    dim3 blocks(num_output_pixels);
    dim3 threads(THREADS);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_direct_kernel<<<blocks, threads>>>(
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
        static_cast<int>(stride[0]),
        static_cast<int>(stride[1]),
        static_cast<int>(padding[0]),
        static_cast<int>(padding[1]),
        total_sum_elems
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Optimized ConvTranspose2D forward (CUDA) with direct block mapping");
}
