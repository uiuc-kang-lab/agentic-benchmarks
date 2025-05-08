#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define SHARED_MEM_SIZE 48*1024

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ float compute_output_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding,
    int b, int oc, int h, int w) {

    float sum = 0.0f;
    int h_offset = h + padding;
    int w_offset = w + padding;
    
    __shared__ float weight_shared[SHARED_MEM_SIZE/sizeof(float)];
    
    int tid = threadIdx.x;
    int weight_elements = K * K;
    for (int i = tid; i < weight_elements; i += BLOCK_SIZE) {
        if (i < weight_elements) {
            weight_shared[i] = weight[oc * weight_elements + i];
        }
    }
    __syncthreads();

    #pragma unroll 4
    for (int ic = 0; ic < C_in; ++ic) {
        int base_input_ic = b * (C_in * H_in * W_in) + ic * (H_in * W_in);
        
        for (int kh = 0; kh < K; kh += 2) {
            int h_in_candidate = h_offset - kh;
            int h_in = h_in_candidate / stride;
            
            if (h_in >= 0 && h_in < H_in && (h_in_candidate % stride) == 0) {
                for (int kw = 0; kw < K; kw += 4) {
                    int w_in_candidate = w_offset - kw;
                    int w_in = w_in_candidate / stride;
                    
                    if (w_in >= 0 && w_in < W_in && (w_in_candidate % stride) == 0) {
                        if (kw + 4 <= K && w_in + 4 <= W_in) {
                            float4 input_vec = load_float4(&input[base_input_ic + h_in * W_in + w_in]);
                            float4 weight_vec = load_float4(&weight_shared[kh * K + kw]);
                            
                            sum += input_vec.x * weight_vec.x;
                            sum += input_vec.y * weight_vec.y;
                            sum += input_vec.z * weight_vec.z;
                            sum += input_vec.w * weight_vec.w;
                        } else {
                            sum += input[base_input_ic + h_in * W_in + w_in] * 
                                  weight_shared[kh * K + kw];
                        }
                    }
                }
            }
        }
    }
    return sum;
}

__global__ void conv_transpose2d_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= H_out * W_out || idy >= B * C_out) return;
    
    int w = idx % W_out;
    int h = idx / W_out;
    int oc = idy % C_out;
    int b = idy / C_out;

    float result = compute_output_element(input, weight,
                                        B, C_in, H_in, W_in,
                                        C_out, H_out, W_out,
                                        K, stride, padding,
                                        b, oc, h, w);

    if (bias != nullptr) {
        result += bias[oc];
    }

    output[b * (C_out * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w] = result;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Input and weight must be CUDA tensors");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    TORCH_CHECK(groups == 1 && output_padding == 0, "Unsupported parameters");

    auto dims = input.sizes();
    int B = dims[0], C_in = dims[1], H_in = dims[2], W_in = dims[3];
    int K = weight.size(2);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    auto output = torch::zeros({B, C_out, H_out, W_out}, input.options());

    dim3 threads(16, 32);
    dim3 blocks(
        (H_out * W_out + threads.x - 1) / threads.x,
        (B * C_out + threads.y - 1) / threads.y
    );

    conv_transpose2d_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding);

    return output;
}