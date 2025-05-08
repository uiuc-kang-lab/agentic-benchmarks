#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define TILE_SIZE 16

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * out_h * out_w) return;
    int oc = (idx / (out_h * out_w)) % out_channels;
    output[idx] = __ldg(&bias[oc]);
}

__global__ void conv_transposed_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels_per_group,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_h,
    const int out_w,
    const int in_channels_per_group) {

    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
    const int threads_per_block = TILE_SIZE * TILE_SIZE;

    int oc_start = blockIdx.y * TILE_SIZE;
    int sp_start = blockIdx.z * TILE_SIZE;

    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    int n = blockIdx.x;

    float sum = 0.0f;

    for (int c_group = 0; c_group < in_channels_per_group; c_group += TILE_SIZE) {
        int c_cur = c_group + threadIdx.y;
        if (c_cur >= in_channels_per_group) continue;

        for (int w_pos = 0; w_pos < kernel_w; w_pos++) {
            __syncthreads();
            
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                int in_row = (sp_start + threadIdx.x) * stride_h - pad_h + kh * dilation_h;
                if (in_row < 0 || in_row >= in_h) continue;
                
                int x_idx = n * in_channels * in_h * in_w
                    + (blockIdx.y * in_channels_per_group + c_cur) * in_h * in_w
                    + in_row * in_w
                    + w_pos;
                
                if (c_group == 0 && w_pos == 0) {
                    tile[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                tile[threadIdx.y][threadIdx.x] = __ldg(&x[x_idx]);
                __syncthreads();

                #pragma unroll
                for (int tw = 0; tw < TILE_SIZE; tw++) {
                    if (c_group + tw < in_channels_per_group) {
                        float weight_val = __ldg(&weight[
                            ((blockIdx.y * in_channels_per_group) + (c_group + tw)) * 
                            (out_channels_per_group * kernel_h * kernel_w) +
                            ((blockIdx.y * out_channels_per_group)) * kernel_h * kernel_w +
                            kh * kernel_w +
                            w_pos
                        ]);
                        sum += tile[tw][threadIdx.x] * weight_val;
                    }
                }
                __syncthreads();
            }
        }
    }

    if (sp_start + threadIdx.x < out_w) {
        int out_idx = n * groups * out_channels_per_group * out_h * out_w
            + (blockIdx.y * out_channels_per_group + oc_start) * out_h * out_w
            + (blockIdx.z * TILE_SIZE + threadIdx.y) * out_w
            + sp_start + threadIdx.x;
            
        atomicAdd(&output[out_idx], sum);
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    const int in_channels_per_group = in_channels / groups;

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int out_h = (in_h - 1)*stride_h - 2*pad_h + dilation_h*(kernel_h-1) +1;
    const int out_w = (in_w - 1)*stride_w - 2*pad_w + dilation_w*(kernel_w-1) +1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    if (!bias.has_value()) {
        bias = at::zeros({out_channels}, x.options());
    }

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(batch, groups, (out_h + TILE_SIZE-1)/TILE_SIZE);

    initialize_output_kernel<<<
        (batch * out_channels * out_h * out_w + 511)/512, 512
    >>>(
        output.data_ptr<float>(),
        bias->data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w
    );

    conv_transposed_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_h,
        out_w,
        in_channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Tiled Transposed Convolution (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
