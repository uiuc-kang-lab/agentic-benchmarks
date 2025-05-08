#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel using shared memory for weight caching and minimal synchronization
// Each block processes one output channel, and threads within the block load the corresponding weight slice into shared memory

__global__ void conv_transpose2d_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
    int padding_h,
    int padding_w) {

    // Determine the output pixel coordinates and the output channel for this block
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;  // each block along z corresponds to one output channel

    // Check output bounds
    if (out_x >= output_width || out_y >= output_height)
        return;

    // Allocate shared memory for the weight slice corresponding to this output channel
    extern __shared__ float sh_weight[]; // size = in_channels * kernel_height * kernel_width
    int total_weight = in_channels * kernel_height * kernel_width;

    // Each thread loads one or more elements of the weight tile
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < total_weight; i += blockDim.x * blockDim.y) {
        int cur_in_ch = i / (kernel_height * kernel_width);
        int rem = i % (kernel_height * kernel_width);
        int kh = rem / kernel_width;
        int kw = rem % kernel_width;
        // Compute the index in the weight tensor (layout: in_channels x out_channels x kH x kW)
        sh_weight[i] = weight[cur_in_ch * out_channels * kernel_height * kernel_width + 
                              out_ch * kernel_height * kernel_width + 
                              kh * kernel_width + kw];
    }
    __syncthreads(); // synchronize to ensure shared memory is loaded

    // For each element in the batch, compute the transposed convolution contribution
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        // Loop over all input channels
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            // Loop over kernel dimensions
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = out_x + padding_w - kw;
                    int in_y = out_y + padding_h - kh;

                    // Check if the computed input indices align with the stride
                    if ((in_x % stride_w == 0) && (in_y % stride_h == 0)) {
                        in_x /= stride_w;
                        in_y /= stride_h;
                        if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                            float input_val = input[b * in_channels * input_height * input_width +
                                                      in_ch * input_height * input_width +
                                                      in_y * input_width + in_x];
                            int weight_index = in_ch * kernel_height * kernel_width + kh * kernel_width + kw;
                            float w = sh_weight[weight_index];
                            sum += input_val * w;
                        }
                    }
                }
            }
        }
        // Write the computed sum to the output tensor
        output[b * out_channels * output_height * output_width +
               out_ch * output_height * output_width +
               out_y * output_width + out_x] = sum;
    }
}

// Host function that sets up the kernel launch
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width  = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    // Configure thread block and grid dimensions
    dim3 threads(16, 16);
    dim3 blocks((output_width + threads.x - 1) / threads.x,
                (output_height + threads.y - 1) / threads.y,
                out_channels);

    // Calculate the required shared memory size (in bytes)
    size_t shared_size = in_channels * kernel_height * kernel_width * sizeof(float);

    // Launch the kernel
    conv_transpose2d_kernel_shared<<<blocks, threads, shared_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
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
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}
