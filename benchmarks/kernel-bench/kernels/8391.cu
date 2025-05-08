#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__inline__ __device__
float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv_transpose2d_kernel_warp(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {

    // Calculate output position
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    const int out_x = blockIdx.x;
    const int out_y = blockIdx.y;
    const int batch_out_ch = blockIdx.z;
    
    const int batch = batch_out_ch / out_channels;
    const int out_ch = batch_out_ch % out_channels;

    if (out_x >= output_width || out_y >= output_height || batch >= batch_size)
        return;

    // Each warp handles a portion of input channels
    float warp_sum = 0.0f;
    
    const int warp_ic_start = (in_channels * (warp_id + 0)) / (BLOCK_SIZE/WARP_SIZE);
    const int warp_ic_end = (in_channels * (warp_id + 1)) / (BLOCK_SIZE/WARP_SIZE);

    #pragma unroll 4
    for (int in_ch = warp_ic_start; in_ch < warp_ic_end; in_ch++) {
        // Each lane in the warp handles different kernel positions
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = lane_id; kw < kernel_width; kw += WARP_SIZE) {
                const int in_x = out_x + pad_w - kw;
                const int in_y = out_y + pad_h - kh;

                if ((in_x % stride_w == 0) && (in_y % stride_h == 0)) {
                    const int mapped_in_x = in_x / stride_w;
                    const int mapped_in_y = in_y / stride_h;

                    if (mapped_in_x >= 0 && mapped_in_x < input_width && 
                        mapped_in_y >= 0 && mapped_in_y < input_height) {
                        
                        const float input_val = input[
                            ((batch * in_channels + in_ch) * input_height + mapped_in_y) * input_width + mapped_in_x
                        ];
                        
                        const float weight_val = weight[
                            ((in_ch * out_channels + out_ch) * kernel_height + kh) * kernel_width + kw
                        ];

                        warp_sum += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Reduce within warp
    warp_sum = warpReduceSum(warp_sum);

    // First thread in each warp writes to shared memory
    __shared__ float warp_results[BLOCK_SIZE/WARP_SIZE];
    if (lane_id == 0) {
        warp_results[warp_id] = warp_sum;
    }
    __syncthreads();

    // First warp reduces results from all warps
    if (warp_id == 0 && lane_id == 0) {
        float final_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE/WARP_SIZE; ++i) {
            final_sum += warp_results[i];
        }

        if (bias) {
            final_sum += bias[out_ch];
        }

        output[
            ((batch * out_channels + out_ch) * output_height + out_y) * output_width + out_x
        ] = final_sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    dim3 blocks(output_width, output_height, batch_size * out_channels);
    dim3 threads(BLOCK_SIZE);

    conv_transpose2d_kernel_warp<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward with warp primitives (CUDA)");
}