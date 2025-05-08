#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_shared(
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
    const int dilation
) {
    extern __shared__ scalar_t shared_data[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / 32;  // warp ID
    const int lane = tid % 32;  // lane within warp
    
    const int output_size = output_height * output_width;
    const int total_outputs = batch_size * channels * output_size;
    
    for (int idx = bid * blockDim.x + tid; idx < total_outputs; idx += gridDim.x * blockDim.x) {
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / output_size) % channels;
        const int b = idx / (channels * output_size);
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Each thread processes its portion of the kernel window
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = ow * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
        
        // Store in shared memory
        shared_data[tid] = max_val;
        __syncthreads();
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const scalar_t other = __shfl_down_sync(0xffffffff, max_val, offset);
            max_val = max(max_val, other);
        }
        
        // First thread in each warp writes the result
        if (lane == 0) {
            output[idx] = max_val;
        }
    }
}

torch::Tensor max_pool2d_cuda_forward(
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

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = std::min(65535, (batch_size * channels * output_height * output_width + threads - 1) / threads);
    const int shared_memory_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory and warp reduction (CUDA)");
}