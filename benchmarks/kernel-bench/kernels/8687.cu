#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int depth,
    const int height,
    const int width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int groups) {
    
    extern __shared__ scalar_t shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * BLOCK_SIZE + tid;
    
    // Calculate output dimensions
    const int out_depth = (depth - 1) * stride_d - 2 * padding_d + kernel_d;
    const int out_height = (height - 1) * stride_h - 2 * padding_h + kernel_h;
    const int out_width = (width - 1) * stride_w - 2 * padding_w + kernel_w;
    
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    for(int i = idx; i < total_elements; i += gridDim.x * BLOCK_SIZE) {
        // Calculate output position
        const int w = i % out_width;
        const int h = (i / out_width) % out_height;
        const int d = (i / (out_width * out_height)) % out_depth;
        const int c = (i / (out_width * out_height * out_depth)) % out_channels;
        const int b = i / (out_width * out_height * out_depth * out_channels);
        
        scalar_t sum = 0;
        
        // Compute convolution
        #pragma unroll 4
        for(int kd = 0; kd < kernel_d; kd++) {
            #pragma unroll 4
            for(int kh = 0; kh < kernel_h; kh++) {
                #pragma unroll 4
                for(int kw = 0; kw < kernel_w; kw++) {
                    if(d-kd >= 0 && h-kh >= 0 && w-kw >= 0 &&
                       d-kd < depth && h-kh < height && w-kw < width) {
                        const int in_idx = b * in_channels * depth * height * width +
                                         c * depth * height * width +
                                         (d-kd) * height * width +
                                         (h-kh) * width +
                                         (w-kw);
                        const int weight_idx = c * kernel_d * kernel_h * kernel_w +
                                             kd * kernel_h * kernel_w +
                                             kh * kernel_w +
                                             kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[i] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }

    return at::conv_transpose3d(
        x,
        weight,
        bias.has_value() ? *bias : at::Tensor(),
        stride,
        padding,
        output_padding,
        groups
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}