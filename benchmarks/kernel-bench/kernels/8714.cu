#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_DIM = 256;  // 8 warps per block

__device__ __inline__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void optimized_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups, int in_channels_per_group) {
    
    const int global_warp_id = blockIdx.x * (blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int total_outputs = N * C_out * outD * outH * outW;
    
    if(global_warp_id >= total_outputs) return;

    int n = global_warp_id / (C_out * outD * outH * outW);
    int remainder = global_warp_id % (C_out * outD * outH * outW);
    int oc = remainder / (outD * outH * outW);
    remainder %= (outD * outH * outW);
    int od = remainder / (outH * outW);
    remainder %= (outH * outW);
    int oh = remainder / outW;
    int ow = remainder % outW;

    const int group = oc / (C_out / groups);
    const int oc_in_group = oc % (C_out / groups);
    const int c_base = group * in_channels_per_group;

    float sum = 0.0f;
    const int ksize = kernel_d * kernel_h * kernel_w;
    
    for(int c_offset = 0; c_offset < in_channels_per_group; ++c_offset) {
        const int c_in = c_base + c_offset;
        
        for(int k_id = lane_id; k_id < ksize; k_id += WARP_SIZE) {
            const int kd = k_id / (kernel_h * kernel_w);
            const int kh = (k_id % (kernel_h * kernel_w)) / kernel_w;
            const int kw = k_id % kernel_w;
            
            const int d_in = od - (kd - pad_d);
            const int h_in = oh - (kh - pad_h);
            const int w_in = ow - (kw - pad_w);

            if(d_in % stride_d || h_in % stride_h || w_in % stride_w)
                continue;
                
            const int d_in_s = d_in / stride_d;
            const int h_in_s = h_in / stride_h;
            const int w_in_s = w_in / stride_w;

            if(d_in_s >= 0 && d_in_s < D_in &&
               h_in_s >= 0 && h_in_s < H_in &&
               w_in_s >= 0 && w_in_s < W_in) {
                
                const int input_idx = (((n * C_in + c_in) * D_in + d_in_s) * H_in + h_in_s) * W_in + w_in_s;
                const int weight_idx = ((((c_base + c_offset) * (C_out/groups) + oc_in_group) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    sum = warp_reduce(sum);
    
    if(lane_id == 0)
        output[global_warp_id] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    const int C_out = weight.size(1) * groups;
    
    const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
    const int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    
    const int outD = (D_in-1)*stride_d - 2*pad_d + kernel_d + output_padding[0];
    const int outH = (H_in-1)*stride_h - 2*pad_h + kernel_h + output_padding[1];
    const int outW = (W_in-1)*stride_w - 2*pad_w + kernel_w + output_padding[2];
    
    auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());
    
    const int total_output = N * C_out * outD * outH * outW;
    const int warps_per_block = BLOCK_DIM / WARP_SIZE;
    const int blocks = (total_output + warps_per_block - 1) / warps_per_block;
    
    optimized_conv_transpose3d_kernel<<<blocks, BLOCK_DIM>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        outD, outH, outW,
        groups, C_in/groups
    );
    
    if(bias.has_value()) {
        auto bias_tensor = *bias;
        CHECK_INPUT(bias_tensor);
        output.add_(bias_tensor.view({1, C_out, 1, 1, 1}));
    }
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized transposed 3D conv with warp reductions");
}