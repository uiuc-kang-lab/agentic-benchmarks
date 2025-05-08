#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float const_weight[8192]; // Adjust size based on maximum expected weight size

__global__ void block_size_optimized_kernel(
    const float* __restrict__ x,
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

    const int TILE_SIZE = 16;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int w_out = blockIdx.x * TILE_SIZE + tx;
    const int h_out = blockIdx.y * TILE_SIZE + ty;
    const int oc = blockIdx.z;

    if (w_out >= width_out || h_out >= height_out || oc >= out_channels) return;

    // Shared memory for input tile
    __shared__ float shared_input[18][18];  // TILE_SIZE + 2 border pixels

    for (int b = 0; b < batch_size; ++b) {
        float sum = bias ? bias[oc] : 0.0f;

        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ++ic) {
            // Load input tile into shared memory
            const int tile_h_start = blockIdx.y * TILE_SIZE * stride - pad_h;
            const int tile_w_start = blockIdx.x * TILE_SIZE * stride - pad_w;

            // Each thread loads one element into shared memory
            for (int i = ty; i < TILE_SIZE + 2; i += TILE_SIZE) {
                for (int j = tx; j < TILE_SIZE + 2; j += TILE_SIZE) {
                    int h_in = tile_h_start + i;
                    int w_in = tile_w_start + j;
                    
                    float val = 0.0f;
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        val = x[((b * in_channels + ic) * input_height + h_in) * input_width + w_in];
                    }
                    shared_input[i][j] = val;
                }
            }
            __syncthreads();

            // Compute convolution using shared memory
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int h_in_local = ty * stride + kh * dilation_h;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int w_in_local = tx * stride + kw * dilation_w;
                    const int w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += shared_input[h_in_local + 1][w_in_local + 1] * const_weight[w_idx];
                }
            }
            __syncthreads();
        }

        const int out_idx = ((b * out_channels + oc) * height_out + h_out) * width_out + w_out;
        output[out_idx] = sum;
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
    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    dim3 threads(16, 16);
    dim3 blocks(
        (width_out + threads.x - 1) / threads.x,
        (height_out + threads.y - 1) / threads.y,
        out_channels
    );

    block_size_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
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