#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t>
__device__ __forceinline__ float4 load_float4(const scalar_t* __restrict__ ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

template<typename scalar_t, int KERNEL_SIZE>
__device__ __forceinline__ scalar_t compute_max_aligned(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int ih_start,
    const int iw_start,
    const int input_height,
    const int input_width,
    const int input_stride_h,
    const int stride,
    const int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int h_offset = ih * input_stride_h;
            const scalar_t* row_ptr = input + base_offset + h_offset;
            
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw += 4) {
                const int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw + 3 * dilation < input_width && 
                    ((reinterpret_cast<std::uintptr_t>(row_ptr + iw) & 15) == 0)) {
                    // Aligned vector load
                    float4 values = load_float4(row_ptr + iw);
                    max_val = max(max_val, values.x);
                    if (kw + 1 < KERNEL_SIZE) max_val = max(max_val, values.y);
                    if (kw + 2 < KERNEL_SIZE) max_val = max(max_val, values.z);
                    if (kw + 3 < KERNEL_SIZE) max_val = max(max_val, values.w);
                } else {
                    // Fallback to scalar loads
                    if (iw >= 0 && iw < input_width)
                        max_val = max(max_val, __ldg(row_ptr + iw));
                    if (kw + 1 < KERNEL_SIZE && iw + dilation >= 0 && iw + dilation < input_width)
                        max_val = max(max_val, __ldg(row_ptr + iw + dilation));
                    if (kw + 2 < KERNEL_SIZE && iw + 2 * dilation >= 0 && iw + 2 * dilation < input_width)
                        max_val = max(max_val, __ldg(row_ptr + iw + 2 * dilation));
                    if (kw + 3 < KERNEL_SIZE && iw + 3 * dilation >= 0 && iw + 3 * dilation < input_width)
                        max_val = max(max_val, __ldg(row_ptr + iw + 3 * dilation));
                }
            }
        }
    }
    return max_val;
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
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int channel_idx = by;
    const int batch_idx = bz;

    if (batch_idx >= batch_size || channel_idx >= channels) return;

    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int input_stride_h = input_width;

    const int base_offset = batch_idx * input_stride_batch + channel_idx * input_stride_channel;

    for (int idx = tx; idx < output_height * output_width; idx += blockDim.x) {
        const int ow = idx % output_width;
        const int oh = idx / output_width;

        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;

        scalar_t max_val;
        
        if (kernel_size == 2) {
            max_val = compute_max_aligned<scalar_t, 2>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                stride, dilation);
        }
        else if (kernel_size == 3) {
            max_val = compute_max_aligned<scalar_t, 3>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                stride, dilation);
        }
        else {
            max_val = compute_max_aligned<scalar_t, 4>(
                input, base_offset, ih_start, iw_start,
                input_height, input_width, input_stride_h,
                stride, dilation);
        }

        const int output_idx = batch_idx * channels * output_height * output_width +
                             channel_idx * output_height * output_width +
                             oh * output_width +
                             ow;
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
    const dim3 blocks(
        1,
        channels,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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