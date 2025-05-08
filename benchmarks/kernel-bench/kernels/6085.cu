#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: combines grid-based thread mapping with conditional loop unrolling
// to optimize runtime efficiency in the common case (fully inside pooling window) and handle
// boundaries gracefully.

template <typename scalar_t>
__global__ void fused_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int kernel_size,
    const int stride,
    const int padding
) {
    // Determine output coordinates via 2D thread mapping
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Use blockIdx.z to cover combined (n, c) dimensions
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    if (out_x >= outW || out_y >= outH) return;

    // Compute top-left coordinates of the pooling window in the input
    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;

    scalar_t sum = scalar_t(0);

    // Check if pooling window lies completely inside the input bounds
    bool fully_inside = (in_x_start >= 0) && (in_y_start >= 0) && 
                        ((in_x_start + kernel_size) <= W) && 
                        ((in_y_start + kernel_size) <= H);

    if (fully_inside) {
        // When fully contained, use loop unrolling for optimal performance
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            int in_y = in_y_start + ky;
            int base = ((n * C + c) * H + in_y) * W;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = in_x_start + kx;
                sum += input[base + in_x];
            }
        }
    } else {
        // For border cases, compute effective window boundaries
        int x_start = (in_x_start < 0) ? 0 : in_x_start;
        int y_start = (in_y_start < 0) ? 0 : in_y_start;
        int x_end = (in_x_start + kernel_size > W) ? W : in_x_start + kernel_size;
        int y_end = (in_y_start + kernel_size > H) ? H : in_y_start + kernel_size;

        for (int y = y_start; y < y_end; y++) {
            int base = ((n * C + c) * H + y) * W;
            for (int x = x_start; x < x_end; x++) {
                sum += input[base + x];
            }
        }
    }

    // Write the output value
    int out_index = ((n * C + c) * outH + out_y) * outW + out_x;
    output[out_index] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Host function for launching the fused kernel

torch::Tensor fused_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Configure a 2D thread block with 3D grid to cover (outW, outH) and the combined (N, C) dimension
    const dim3 threads(32, 8);
    const dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_avg_pool2d_kernel", ([&] {
        fused_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_cont.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_avg_pool2d_forward, "Fused Optimized 2D Average Pooling forward (CUDA)");
}
