#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements a custom transposed convolution for square input and kernel
// using shared memory for intra-block reduction and warp-level primitives for the final reduction stage.
// It supports groups==1 only.

// Kernel: Each block computes one output element (n, c_out, h, w).
// Threads within the block collaboratively reduce the partial sums computed over the input channels and kernel elements.

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,    // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,   // [C_in, C_out, K, K]
    const float* __restrict__ bias,     // [C_out] or nullptr
    float* __restrict__ output,         // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K,
    int H_out, int W_out,
    int stride, int padding, int output_padding
) {
    // Each block is responsible for one output element
    int index = blockIdx.x; // index in [0, N * C_out * H_out * W_out)
    int out_area = H_out * W_out;
    int n = index / (C_out * out_area);
    int rem = index % (C_out * out_area);
    int c_out = rem / out_area;
    int pos = rem % out_area;
    int h = pos / W_out;
    int w = pos % W_out;

    // Shared memory for reduction
    extern __shared__ float sdata[]; // size: blockDim.x * sizeof(float)

    float partial = 0.0f;
    // Each thread loops over a subset of input channels (c) with stride = blockDim.x
    for (int c = threadIdx.x; c < C_in; c += blockDim.x) {
        // Loop over the kernel spatial dimensions
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                // Compute the corresponding input spatial coordinate
                int h_temp = h + padding - ki;
                int w_temp = w + padding - kj;
                // The input pixel contributes only if h_temp and w_temp are divisible by the stride
                if ((h_temp % stride == 0) && (w_temp % stride == 0)) {
                    int h_in = h_temp / stride;
                    int w_in = w_temp / stride;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Compute indices
                        int input_idx = ((n * C_in + c) * H_in + h_in) * W_in + w_in;
                        int weight_idx = (((c * C_out + c_out) * K) + ki) * K + kj;
                        partial += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Store the thread's partial sum in shared memory
    sdata[threadIdx.x] = partial;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for the final 32 values
    float sum = sdata[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first thread writes the final result
    if (threadIdx.x == 0) {
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        int out_idx = ((n * C_out + c_out) * H_out + h) * W_out + w;
        output[out_idx] = sum;
    }
}

// Forward function definition

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    // This implementation supports groups == 1 only
    TORCH_CHECK(groups == 1, "shmem_reduction_conv_transpose2d only supports groups == 1");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    // Input dimensions: [N, C_in, H_in, W_in]
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Weight dimensions: [C_in, C_out, K, K] (square kernel assumed)
    TORCH_CHECK(weight.dim() == 4, "Weight tensor must be 4D");
    int C_out = weight.size(1);
    int K = weight.size(2);

    // Compute output spatial dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    // Allocate output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Each block computes one output element, so total number of blocks:
    int total_blocks = N * C_out * H_out * W_out;
    int block_size = 128; // Tune this value if needed
    size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    conv_transpose2d_kernel<<<total_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K,
        H_out, W_out,
        stride, padding, output_padding
    );
    cudaDeviceSynchronize();

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - shared memory reduction");
}
