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
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int output_idx = blockIdx.x * block_size + tid;
    
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    // Calculate input region boundaries
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;
    const int ih_end = min(ih_start + kernel_size * dilation, input_height);
    const int iw_end = min(iw_start + kernel_size * dilation, input_width);

    // Calculate shared memory dimensions for this block
    const int sh_width = (kernel_size * dilation + stride - 1);
    const int sh_height = (kernel_size * dilation + stride - 1);
    
    // Load input region into shared memory
    for (int i = tid; i < sh_height * sh_width; i += block_size) {
        int sh_h = i / sh_width;
        int sh_w = i % sh_width;
        int ih = ih_start + sh_h;
        int iw = iw_start + sh_w;
        
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            shared_input[sh_h * sh_width + sh_w] = input[b * (channels * input_height * input_width) +
                                                        c * (input_height * input_width) +
                                                        ih * input_width + iw];
        } else {
            shared_input[sh_h * sh_width + sh_w] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
    
    __syncthreads();

    // Compute max pooling from shared memory
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            const int sh_h = kh * dilation;
            const int sh_w = kw * dilation;
            if (sh_h < sh_height && sh_w < sh_width) {
                max_val = max(max_val, shared_input[sh_h * sh_width + sh_w]);
            }
        }
    }

    if (output_idx < batch_size * channels * output_height * output_width) {
        output[output_idx] = max_val;
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
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    
    // Calculate shared memory size
    const int sh_width = (kernel_size * dilation + stride - 1);
    const int sh_height = (kernel_size * dilation + stride - 1);
    const int shared_memory_size = sh_width * sh_height * sizeof(float);

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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}