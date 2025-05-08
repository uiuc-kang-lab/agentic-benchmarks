#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void strided_conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride, const int padding,
    const int H_out, const int W_out) {
    
    // Calculate total threads and strides
    const int thread_count = blockDim.x * gridDim.x;
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total output elements
    const int total_outputs = N * C_out * H_out * W_out;
    
    // Stride loop over output elements
    for (int idx = thread_idx; idx < total_outputs; idx += thread_count) {
        // Decode output index
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c_out = (idx / (W_out * H_out)) % C_out;
        const int n = idx / (W_out * H_out * C_out);
        
        float sum = 0.0f;
        
        // Calculate input window boundaries
        const int h_start = (h_out + padding) / stride;
        const int w_start = (w_out + padding) / stride;
        const int h_end = min((h_out + padding + K - 1) / stride + 1, H);
        const int w_end = min((w_out + padding + K - 1) / stride + 1, W);
        
        // Loop over input window
        for (int h_in = h_start; h_in < h_end; h_in++) {
            const int kh = h_out + padding - h_in * stride;
            if (kh < 0 || kh >= K) continue;
            
            for (int w_in = w_start; w_in < w_end; w_in++) {
                const int kw = w_out + padding - w_in * stride;
                if (kw < 0 || kw >= K) continue;
                
                // Accumulate contributions from all input channels
                for (int c_in = 0; c_in < C_in; c_in++) {
                    const float in_val = input[((n * C_in + c_in) * H + h_in) * W + w_in];
                    const float w_val = weight[((c_in * C_out + c_out) * K + kh) * K + kw];
                    sum += in_val * w_val;
                }
            }
        }
        
        output[idx] = sum;
    }
}

__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    const int N, const int C_out,
    const int H_out, const int W_out) {
    
    const int thread_count = blockDim.x * gridDim.x;
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C_out * H_out * W_out;
    
    for (int idx = thread_idx; idx < total_elements; idx += thread_count) {
        const int c = (idx / (H_out * W_out)) % C_out;
        output[idx] += bias[c];
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "Weight must be a contiguous CUDA tensor");
    
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int C_out = weight.size(1);
    const int K = weight.size(2);
    
    const int H_out = (H - 1) * stride - 2 * padding + K + output_padding;
    const int W_out = (W - 1) * stride - 2 * padding + K + output_padding;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    // Configure kernel launch parameters
    const int block_size = 256;
    const int num_blocks = min(65535, (N * C_out * H_out * W_out + block_size - 1) / block_size);
    
    strided_conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K,
        stride, padding,
        H_out, W_out
    );
    
    if (bias.has_value()) {
        add_bias_kernel<<<num_blocks, block_size>>>(
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            N, C_out, H_out, W_out
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}