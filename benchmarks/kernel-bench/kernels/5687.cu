#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Declare constant memory arrays for pooling window offsets
// For a 2x2 pooling window (4 elements, 8 integers) and a 3x3 pooling window (9 elements, 18 integers)
__constant__ int pool_offsets_2[8];    // Format: {0,0, 0,1, 1,0, 1,1}
__constant__ int pool_offsets_3[18];   // Format: {0,0, 0,1, 0,2, 1,0, 1,1, 1,2, 2,0, 2,1, 2,2}

// Kernel for kernel_size == 2 using constant memory for offsets
template <typename scalar_t>
__global__ void max_pool2d_const_kernel_2(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation) {

    int total = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    while (idx < total) {
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int b = idx / (output_width * output_height * channels);
        int input_offset = b * channels * input_height * input_width + c * input_height * input_width;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        // Loop over 2x2 pooling window using offsets from constant memory
        for (int i = 0; i < 4; i++) {
            int dy = pool_offsets_2[2 * i];
            int dx = pool_offsets_2[2 * i + 1];
            int ih = oh * stride - padding + dy * dilation;
            int iw = ow * stride - padding + dx * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                scalar_t val = __ldg(&input[input_offset + ih * input_width + iw]);
                max_val = (val > max_val) ? val : max_val;
            }
        }
        output[idx] = max_val;
        idx += gridSize;
    }
}

// Kernel for kernel_size == 3 using constant memory for offsets
template <typename scalar_t>
__global__ void max_pool2d_const_kernel_3(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation) {

    int total = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    while (idx < total) {
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int b = idx / (output_width * output_height * channels);
        int input_offset = b * channels * input_height * input_width + c * input_height * input_width;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        // Loop over 3x3 pooling window using constant memory offsets
        for (int i = 0; i < 9; i++) {
            int dy = pool_offsets_3[2 * i];
            int dx = pool_offsets_3[2 * i + 1];
            int ih = oh * stride - padding + dy * dilation;
            int iw = ow * stride - padding + dx * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                scalar_t val = __ldg(&input[input_offset + ih * input_width + iw]);
                max_val = (val > max_val) ? val : max_val;
            }
        }
        output[idx] = max_val;
        idx += gridSize;
    }
}

// Generic kernel for arbitrary kernel sizes (without using constant memory for offsets)
template <typename scalar_t>
__global__ void max_pool2d_generic_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    int total = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    while (idx < total) {
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int b = idx / (output_width * output_height * channels);
        int input_offset = b * channels * input_height * input_width + c * input_height * input_width;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < kernel_size; i++) {
            int ih = oh * stride - padding + i * dilation;
            if (ih < 0 || ih >= input_height) continue;
            for (int j = 0; j < kernel_size; j++) {
                int iw = ow * stride - padding + j * dilation;
                if (iw < 0 || iw >= input_width) continue;
                scalar_t val = input[input_offset + ih * input_width + iw];
                max_val = (val > max_val) ? val : max_val;
            }
        }
        output[idx] = max_val;
        idx += gridSize;
    }
}

// Host function launching the appropriate kernel
torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            // Prepare constant memory for 2x2 pooling window offsets: {0,0, 0,1, 1,0, 1,1}
            int h_offsets[8] = {0, 0, 0, 1, 1, 0, 1, 1};
            cudaMemcpyToSymbol(pool_offsets_2, h_offsets, sizeof(h_offsets));
            max_pool2d_const_kernel_2<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width,
                stride, padding, dilation);
        } else if (kernel_size == 3) {
            // Prepare constant memory for 3x3 pooling window offsets: {0,0, 0,1, 0,2, 1,0, 1,1, 1,2, 2,0, 2,1, 2,2}
            int h_offsets[18] = {0,0, 0,1, 0,2, 1,0, 1,1, 1,2, 2,0, 2,1, 2,2};
            cudaMemcpyToSymbol(pool_offsets_3, h_offsets, sizeof(h_offsets));
            max_pool2d_const_kernel_3<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width,
                stride, padding, dilation);
        } else {
            // Fallback to generic kernel if kernel_size is not 2 or 3
            max_pool2d_generic_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width,
                kernel_size, stride, padding, dilation);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with constant memory optimized offsets (CUDA)");
}
