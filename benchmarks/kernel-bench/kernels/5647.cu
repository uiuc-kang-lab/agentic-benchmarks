#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define VEC_WIDTH 4

template <typename scalar_t>
__global__ void vectorized_maxpool2d_kernel(
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
    const int total = batch_size * channels * output_height * output_width;
    const int vec_total = total / VEC_WIDTH;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;

    typedef typename cuda::type::vec<scalar_t, VEC_WIDTH>::type VecType;
    
    for (; idx < vec_total; idx += grid_stride) {
        const int base_idx = idx * VEC_WIDTH;
        VecType max_vals;
        for (int v = 0; v < VEC_WIDTH; v++) {
            max_vals[v] = -std::numeric_limits<scalar_t>::infinity();
        }

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int elem_base = base_idx;
                for (int v = 0; v < VEC_WIDTH; v++) {
                    const int out_idx = elem_base + v;
                    if (out_idx >= total) break;

                    const int ow = out_idx % output_width;
                    const int oh = (out_idx / output_width) % output_height;
                    const int c = (out_idx / (output_width * output_height)) % channels;
                    const int b = out_idx / (output_width * output_height * channels);

                    const int ih_v = oh * stride - padding + kh * dilation;
                    const int iw_v = ow * stride - padding + kw * dilation;
                    
                    if (ih_v >= 0 && ih_v < input_height && iw_v >= 0 && iw_v < input_width) {
                        const int input_idx = b * channels * input_height * input_width
                                          + c * input_height * input_width
                                          + ih_v * input_width + iw_v;
                        scalar_t val = input[input_idx];
                        if (val > max_vals[v]) max_vals[v] = val;
                    }
                }
            }
        }

        for (int v = 0; v < VEC_WIDTH; v++) {
            const int out_idx = base_idx + v;
            if (out_idx < total) output[out_idx] = max_vals[v];
        }
    }

    int res_idx = vec_total * VEC_WIDTH + blockIdx.x * blockDim.x + threadIdx.x;
    if (res_idx < total) {
        const int ow = res_idx % output_width;
        const int oh = (res_idx / output_width) % output_height;
        const int c = (res_idx / (output_width * output_height)) % channels;
        const int b = res_idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = b * channels * input_height * input_width
                                     + c * input_height * input_width
                                     + ih * input_width + iw;
                    scalar_t val = input[input_idx];
                    max_val = max(max_val, val);
                }
            }
        }
        output[res_idx] = max_val;
    }
}

torch::Tensor vectorized_max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int vec_total = (total + VEC_WIDTH - 1) / VEC_WIDTH;
    const int blocks = (vec_total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vectorized_max_pool2d_cuda_forward", ([&] {
        vectorized_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vectorized_max_pool2d_cuda_forward, "Vectorized Max Pool 2D forward (CUDA)");
}
