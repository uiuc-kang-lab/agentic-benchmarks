#include <torch/extension.h>

template<int K>
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int stride, int padding, int output_padding,
    int groups) {

    extern __shared__ float s_weights[];
    const int c_per_group_out = C_out / groups;
    const int c_per_group_in = C_in / groups;
    
    const int group = blockIdx.z;
    const int bc = blockIdx.x * blockDim.y + threadIdx.y;
    const int h_out = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (h_out >= H_out) return;
    
    // Load weights into shared memory
    const int weight_load_idx = threadIdx.y * c_per_group_in * K * K + threadIdx.x;
    if (weight_load_idx < c_per_group_in * K * K * c_per_group_out) {
        int c_out_idx = weight_load_idx / (c_per_group_in * K * K);
        int residual = weight_load_idx % (c_per_group_in * K * K);
        int c_in_idx = residual / (K * K);
        int kh = (residual % (K * K)) / K;
        int kw = residual % K;
        s_weights[c_out_idx * c_per_group_in * K * K + c_in_idx * K * K + kh * K + kw] = 
            weight[(group * c_per_group_in + c_in_idx) * C_out * K * K + (group * c_per_group_out + c_out_idx) * K * K + kh * K + kw];
    }
    __syncthreads();

    // Main computation
    for (int b = 0; b < B; ++b) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            float val = 0.0f;
            for (int c_in = 0; c_in < c_per_group_in; ++c_in) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int h_in = (h_out - kh + padding) / stride;
                        int w_in = (w_out - kw + padding) / stride;
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in
                            && (h_out - kh + padding) % stride == 0
                            && (w_out - kw + padding) % stride == 0) {
                            float x_val = x[b * C_in * H_in * W_in + 
                                (group * c_per_group_in + c_in) * H_in * W_in + 
                                h_in * W_in + w_in];
                            int weight_idx = bc * c_per_group_in * K * K + c_in * K * K + kh * K + kw;
                            val += x_val * s_weights[weight_idx];
                        }
                    }
                }
            }
            output[b * C_out * H_out * W_out + 
                (group * c_per_group_out + bc) * H_out * W_out + 
                h_out * W_out + w_out] = val;
        }
    }
}

torch::Tensor conv_transpose2d_forward(torch::Tensor x, torch::Tensor weight, int stride, int padding, int output_padding, int groups) {
    // Ensure the input tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();

    // Get input dimensions
    int B = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Get weight dimensions
    int C_out = weight.size(0);
    int K = weight.size(2);

    // Calculate output dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    // Prepare output tensor
    auto output = torch::zeros({B, C_out, H_out, W_out}, x.options());

    // Launch kernel
    int threads = 16;
    dim3 blockDim(threads, threads);
    dim3 gridDim((C_out + threads - 1) / threads, (H_out + threads - 1) / threads, groups);
    size_t shared_mem_size = C_out * C_in * K * K * sizeof(float) / groups;

    conv_transpose2d_kernel<K><<<gridDim, blockDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride, padding, output_padding,
        groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}