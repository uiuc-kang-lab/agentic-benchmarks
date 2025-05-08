#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int d_params[4];  // kernel_size, stride, padding, dilation

// Unified kernel with shared memory usage and constant memory parameters
template <typename scalar_t>
__global__ void unified_max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    extern __shared__ scalar_t shared_input[];  // Shared memory for input tiles

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int c = blockIdx.z * blockDim.z + tz;

    if (c >= channels || ow >= output_width || oh >= output_height) return;

    const int b = blockIdx.w;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Load data into shared memory with guard
    for (int kh = 0; kh < d_params[0]; kh++) {
        for (int kw = 0; kw < d_params[0]; kw++) {
            const int ih = oh * d_params[1] - d_params[2] + kh * d_params[3];
            const int iw = ow * d_params[1] - d_params[2] + kw * d_params[3];

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                      c * (input_height * input_width) +
                                      ih * input_width +
                                      iw;
                shared_input[threadIdx.y * blockDim.x + threadIdx.x] = __ldg(&input[input_idx]);
            }
            __syncthreads();
            
            // Compute max value
            max_val = max(max_val, shared_input[threadIdx.y * blockDim.x + threadIdx.x]);
        }
    }

    const int output_idx = b * (channels * output_height * output_width) +
                           c * (output_height * output_width) +
                           oh * output_width +
                           ow;

    output[output_idx] = max_val;
}

// Host function
torch::Tensor unified_max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Copy parameters to constant memory
    int h_params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(d_params, h_params, sizeof(int) * 4);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Configure block size and number
    const dim3 threads(16, 16, 4);  // 3D thread block
    const dim3 blocks((output_width + threads.x - 1) / threads.x,
                      (output_height + threads.y - 1) / threads.y,
                      (channels + threads.z - 1) / threads.z,
                      batch_size);

    size_t shared_mem_size = threads.x * threads.y * sizeof(scalar_t);  // Calculate shared memory size needed

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unified_max_pool2d_cuda_forward", ([&] {
        unified_max_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unified_max_pool2d_cuda_forward, "Unified Max Pool 2D forward (CUDA)");
}