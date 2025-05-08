#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__device__ __forceinline__ void compute_indices(int linear_idx, int output_width, int output_height, int channels, int& b, int& c, int& oh, int& ow) {
    ow = linear_idx % output_width;
    oh = (linear_idx / output_width) % output_height;
    c = (linear_idx / (output_width * output_height)) % channels;
    b = linear_idx / (output_width * output_height * channels);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_window_max(const scalar_t* __restrict__ input,
                                                    int b, int c, int oh, int ow,
                                                    int input_h, int input_w,
                                                    int kernel, int stride,
                                                    int pad, int dilation) {
    const int base_h = oh * stride - pad;
    const int base_w = ow * stride - pad;
    const int plane_size = input_h * input_w;
    const int base_offset = b * plane_size * channels + c * plane_size;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

#pragma unroll 3
    for (int kh = 0; kh < kernel; ++kh) {
        const int ih = base_h + kh * dilation;
        if (ih < 0 || ih >= input_h) continue;
        
#pragma unroll 3
        for (int kw = 0; kw < kernel; ++kw) {
            const int iw = base_w + kw * dilation;
            if (iw < 0 || iw >= input_w) continue;
            
            const scalar_t val = input[base_offset + ih * input_w + iw];
            max_val = max(max_val, val);
        }
    }
    return max_val;
}

template <typename scalar_t>
__global__ void tiled_maxpool2d_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                                    const int B, const int C, const int H, const int W,
                                    const int OH, const int OW, const int K,
                                    const int S, const int P, const int D) {
    const int total = B * C * OH * OW;
    const int G = gridDim.x * blockDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += G) {
        int b, c, oh, ow;
        compute_indices<scalar_t>(idx, OW, OH, C, b, c, oh, ow);
        output[idx] = compute_window_max(input, b, c, oh, ow, H, W, K, S, P, D);
    }
}

torch::Tensor tiled_maxpool2d_forward(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    const int OH = (H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1;
    const int OW = (W + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1;

    auto output = torch::empty({B, C, OH, OW}, input.options());

    const int total = B*C*OH*OW;
    const int threads = 256;
    const int blocks = (total + threads-1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tiled_maxpool2d", ([&] {
        tiled_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, C, H, W, OH, OW,
            kernel_size, stride, padding, dilation
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tiled_maxpool2d_forward, "Tiled MaxPool2D forward (CUDA)");
}