#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[4];

// Optimized kernel for balanced workload distribution
template <typename scalar_t>
__global__ void optimized_max_pool_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pool_regions = blockDim.y * gridDim.y;

    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];

    const int regions_per_thread = (output_height * output_width + total_pool_regions - 1) / total_pool_regions;
    const int start_region = (tid / output_width) * regions_per_thread;
    const int end_region = min(start_region + regions_per_thread, output_height);
    const int c = blockIdx.z;

    if (tid >= output_width * output_height) return;
    for (int b = 0; b < blockIdx.y; b++) {
        for (int oh = start_region; oh < end_region; ++oh) {
            const int ow = tid % output_width;
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int ih = oh * stride - padding + kh;
                    const int iw = ow * stride - padding + kw;

                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        const int input_idx = blockIdx.y * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width +
                                            iw;
                        max_val = max(max_val, input[input_idx]);
                    }
                }
            }

            const int output_idx = blockIdx.y * (channels * output_height * output_width) +
                                  c * (output_height * output_width) +
                                  oh * output_width +
                                  ow;
            output[output_idx] = max_val;
        }
    }
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int params[3] = {kernel_size, stride, padding};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 3);

    const dim3 threads(128, 1);  // Focusing on balanced workload
    const dim3 blocks((output_width + threads.x - 1) / threads.x, batch_size, channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        optimized_max_pool_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}