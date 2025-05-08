#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses shared memory to reduce global memory access latency

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int outH,
    int outW,
    int kernel_size,
    int stride,
    int padding
) {
    extern __shared__ scalar_t shared_input[];

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_out >= outW || h_out >= outH) return;

    int c = blockIdx.z % C;
    int n = blockIdx.z / C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t sum_val = scalar_t(0);

    // Load data into shared memory
    for (int i = threadIdx.y; i < kernel_size; i += blockDim.y) {
        for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                shared_input[i * kernel_size + j] = input[((n * C + c) * H + h_in) * W + w_in];
            } else {
                shared_input[i * kernel_size + j] = 0;
            }
        }
    }
    __syncthreads();

    // Compute the average using shared memory
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            sum_val += shared_input[i * kernel_size + j];
        }
    }

    output[((n * C + c) * outH + h_out) * outW + w_out] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
}

// Host function that sets up the kernel launch

torch::Tensor avg_pool2d_forward_shared(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, N * C);

    size_t shared_mem_size = kernel_size * kernel_size * x.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_shared", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_shared<scalar_t><<<grid, block, shared_mem_size>>>(
            input_data,
            output_data,
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward_shared, "2D Average Pooling forward with shared memory (CUDA)");
}
