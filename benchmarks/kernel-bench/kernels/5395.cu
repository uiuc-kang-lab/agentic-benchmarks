#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>


// Optimized kernel: Each block computes one output element using cooperative reduction.
// Threads in the block load portions of the pooling window and perform a warp-level reduction,
// followed by a shared memory reduction across warps using __shfl_down_sync().

template <typename scalar_t>
__global__ void optimized_max_pool2d_kernel(
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
    // Each block is responsible for one output element
    int out_idx = blockIdx.x;
    if (out_idx >= batch_size * channels * output_height * output_width) return;

    // Compute output indices from the linear index
    int ow = out_idx % output_width;
    int oh = (out_idx / output_width) % output_height;
    int c = (out_idx / (output_width * output_height)) % channels;
    int b = out_idx / (output_width * output_height * channels);

    // Compute the base index for the input
    int input_base = b * (channels * input_height * input_width) + c * (input_height * input_width);

    // Determine the starting coordinates for the pooling window
    int input_row_start = oh * stride - padding;
    int input_col_start = ow * stride - padding;

    int pool_elems = kernel_size * kernel_size;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    // Initialize the local maximum to the smallest possible value
    scalar_t local_max = std::numeric_limits<scalar_t>::lowest();

    // Each thread processes a subset of the pooling window
    for (int i = tid; i < pool_elems; i += blockSize) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int row = input_row_start + kh * dilation;
        int col = input_col_start + kw * dilation;
        
        // Check boundaries
        if (row >= 0 && row < input_height && col >= 0 && col < input_width) {
            int input_index = input_base + row * input_width + col;
            local_max = max(local_max, input[input_index]);
        }
    }

    // Perform warp-level reduction using warp primitives
    unsigned int warp_mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(warp_mask, local_max, offset));
    }

    // Use shared memory to reduce across warps
    extern __shared__ char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);

    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_data[warpId] = local_max;
    }
    __syncthreads();

    int numWarps = (blockSize + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        scalar_t val = shared_data[threadIdx.x];
        unsigned int red_mask = __activemask();
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val = max(val, __shfl_down_sync(red_mask, val, offset));
        }
        if (threadIdx.x == 0) {
            output[out_idx] = val;
        }
    }
}


// Forward function that sets up kernel launch parameters
// It chooses the number of threads per block based on the pooling window size to avoid launching
// an excessive number of idle threads when the pooling window is small.

torch::Tensor optimized_max_pool2d_cuda_forward(
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

    int total_output = batch_size * channels * output_height * output_width;
    // Determine number of threads based on pooling window size
    int pool_elems = kernel_size * kernel_size;
    const int threads = (pool_elems < 256) ? pool_elems : 256;
    const int blocks = total_output;  // One block per output element

    int numWarps = (threads + 31) / 32;
    size_t shared_mem_size = numWarps * sizeof(float); // will be replaced by sizeof(scalar_t) below

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_max_pool2d_cuda_forward", ([&] {
        size_t shared_size = numWarps * sizeof(scalar_t);
        optimized_max_pool2d_kernel<scalar_t><<<blocks, threads, shared_size>>>(
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
    m.def("forward", &optimized_max_pool2d_cuda_forward, "Optimized Max Pool 2D forward (CUDA)");
}
