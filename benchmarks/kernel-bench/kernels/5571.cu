#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[8];

template <typename scalar_t>
__device__ __forceinline__ scalar_t fetch_input(const scalar_t* __restrict__ input, int idx) {
    return __ldg(&input[idx]);
}

template <typename scalar_t>
__device__ __forceinline__ void fetch_input4(const scalar_t* __restrict__ input, int idx, scalar_t& v0, scalar_t& v1, scalar_t& v2, scalar_t& v3) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        float4 tmp = *reinterpret_cast<const float4*>(&input[idx]);
        v0 = tmp.x;
        v1 = tmp.y;
        v2 = tmp.z;
        v3 = tmp.w;
    } else {
        v0 = __ldg(&input[idx]);
        v1 = __ldg(&input[idx + 1]);
        v2 = __ldg(&input[idx + 2]);
        v3 = __ldg(&input[idx + 3]);
    }
}

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * output_height * output_width;
    const int vector_elements = total_elements / 4 * 4;

    for (int idx = tid * 4; idx < vector_elements; idx += blockDim.x * gridDim.x * 4) {
        scalar_t max_vals[4] = {
            -std::numeric_limits<scalar_t>::infinity(),
            -std::numeric_limits<scalar_t>::infinity(),
            -std::numeric_limits<scalar_t>::infinity(),
            -std::numeric_limits<scalar_t>::infinity()
        };

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int output_idx = idx + i;
            const int ow = output_idx % output_width;
            const int oh = (output_idx / output_width) % output_height;
            const int c = (output_idx / (output_width * output_height)) % channels;
            const int b = output_idx / (output_width * output_height * channels);

            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                const int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            const int input_idx = b * (channels * input_height * input_width) +
                                                c * (input_height * input_width) +
                                                ih * input_width +
                                                iw;
                            max_vals[i] = max(max_vals[i], fetch_input(input, input_idx));
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same<scalar_t, float>::value && (idx % 4 == 0)) {
            float4* out_ptr = reinterpret_cast<float4*>(&output[idx]);
            float4 result;
            result.x = max_vals[0];
            result.y = max_vals[1];
            result.z = max_vals[2];
            result.w = max_vals[3];
            *out_ptr = result;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                output[idx + i] = max_vals[i];
            }
        }
    }

    for (int idx = vector_elements + tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        const int input_idx = b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width +
                                            iw;
                        max_val = max(max_val, fetch_input(input, input_idx));
                    }
                }
            }
        }
        output[idx] = max_val;
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

    const int params[8] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 8);

    const int threads = 256;
    const int blocks = std::min(65535, (batch_size * channels * output_height * output_width + threads * 4 - 1) / (threads * 4));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}