#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ inline void precompute_valid_indices(
    int stride, int padding, int h, int w,
    int H_in, int W_in, int K,
    int* valid_h, int* valid_w, int* num_valid) {
    
    *num_valid = 0;
    int h_offset = h + padding;
    int w_offset = w + padding;
    
    // Pre-compute all valid indices
    for (int kh = 0; kh < K; kh++) {
        int h_in_candidate = h_offset - kh;
        bool h_valid = (h_in_candidate % stride == 0);
        int h_in = h_in_candidate / stride;
        h_valid &= (h_in >= 0 && h_in < H_in);
        
        for (int kw = 0; kw < K; kw++) {
            int w_in_candidate = w_offset - kw;
            bool w_valid = (w_in_candidate % stride == 0);
            int w_in = w_in_candidate / stride;
            w_valid &= (w_in >= 0 && w_in < W_in);
            
            if (h_valid && w_valid) {
                valid_h[*num_valid] = h_in;
                valid_w[*num_valid] = w_in;
                (*num_valid)++;
            }
        }
    }
}

__device__ inline float compute_output_element_aligned(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K,
    int b, int oc,
    const int* valid_h, const int* valid_w, int num_valid) {
    
    float sum = 0.0f;
    int base_input = b * (C_in * H_in * W_in);
    int weight_oc_offset = oc * (K * K);
    
    // Process all valid indices
    #pragma unroll 4
    for (int idx = 0; idx < num_valid; idx++) {
        int h_in = valid_h[idx];
        int w_in = valid_w[idx];
        int kh = idx / K;
        int kw = idx % K;
        
        // Process all input channels with coalesced access
        for (int ic = 0; ic < C_in; ic++) {
            int input_idx = base_input + ic * (H_in * W_in) + h_in * W_in + w_in;
            int weight_idx = ic * (C_out * K * K) + weight_oc_offset + kh * K + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

__global__ void conv_transpose2d_kernel_warp_aligned(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {
    
    // Shared memory for valid indices
    __shared__ int valid_h[BLOCK_SIZE * K * K];
    __shared__ int valid_w[BLOCK_SIZE][K*K];
    __shared__ int num_valid[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int total_warps = gridDim.x * (blockDim.x / WARP_SIZE);
    int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    
    // Process output elements in a warp-aligned manner
    for (int warp_offset = global_warp_id; warp_offset < B * C_out * (H_out/WARP_SIZE); warp_offset += total_warps) {
        int b = warp_offset / (C_out * (H_out/WARP_SIZE));
        int tmp = warp_offset % (C_out * (H_out/WARP_SIZE));
        int oc = tmp / (H_out/WARP_SIZE);
        int h_base = (tmp % (H_out/WARP_SIZE)) * WARP_SIZE;
        
        // Process WARP_SIZE elements in the width dimension
        for (int w = 0; w < W_out; w++) {
            int h = h_base + lane_id;
            if (h < H_out) {
                // Precompute valid indices for this output position
                precompute_valid_indices(stride, padding, h, w,
                                      H_in, W_in, K,
                                      valid_h[tid], valid_w[tid], &num_valid[tid]);
                
                float sum = compute_output_element_aligned(
                    input, weight,
                    B, C_in, H_in, W_in,
                    C_out, K,
                    b, oc,
                    valid_h[tid], valid_w[tid], num_valid[tid]);
                
                if (bias != nullptr) {
                    sum += bias[oc];
                }
                
                int out_idx = b * (C_out * H_out * W_out) +
                             oc * (H_out * W_out) +
                             h * W_out + w;
                output[out_idx] = sum;
            }
        }
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    TORCH_CHECK(groups == 1, "Only groups==1 is supported");
    TORCH_CHECK(output_padding == 0, "Only output_padding==0 is supported");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }
    
    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int K = weight.size(2);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;
    
    auto output = torch::zeros({B, C_out, H_out, W_out}, input.options());
    
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int num_blocks = (B * C_out * (H_out/WARP_SIZE) + warps_per_block - 1) / warps_per_block;
    
    conv_transpose2d_kernel_warp_aligned<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Warp-aligned ConvTranspose2d forward (CUDA)");
}