#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 256;
const int SHARED_MEM_SIZE = 128; // Number of output elements to process per block

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out, const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int outD, const int outH, const int outW,
    const int groups, const int in_channels_per_group) {
    
    extern __shared__ float shared_output[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    
    // Clear shared memory
    for (int i = tid; i < SHARED_MEM_SIZE; i += num_threads) {
        shared_output[i] = 0.0f;
    }
    __syncthreads();
    
    // Calculate which input elements this block processes
    const int elements_per_block = (N * C_in * D_in * H_in * W_in + gridDim.x - 1) / gridDim.x;
    const int start_idx = bid * elements_per_block;
    const int end_idx = min(start_idx + elements_per_block, N * C_in * D_in * H_in * W_in);
    
    // Process input elements assigned to this block
    for (int idx = start_idx + tid; idx < end_idx; idx += num_threads) {
        // Decode input index
        int w_in = idx % W_in;
        int tmp = idx / W_in;
        int h_in = tmp % H_in;
        tmp /= H_in;
        int d_in = tmp % D_in;
        tmp /= D_in;
        int c_in = tmp % C_in;
        int n = tmp / C_in;
        
        const int group = c_in / in_channels_per_group;
        const float in_val = input[idx];
        
        // Calculate output contribution for this input element
        for (int kd = 0; kd < kernel_d; kd++) {
            const int out_d = d_in * stride_d - pad_d + kd;
            if (out_d < 0 || out_d >= outD) continue;
            
            for (int kh = 0; kh < kernel_h; kh++) {
                const int out_h = h_in * stride_h - pad_h + kh;
                if (out_h < 0 || out_h >= outH) continue;
                
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int out_w = w_in * stride_w - pad_w + kw;
                    if (out_w < 0 || out_w >= outW) continue;
                    
                    // Process output channels for this group
                    for (int oc = 0; oc < C_out / groups; oc++) {
                        const int oc_global = group * (C_out / groups) + oc;
                        
                        // Calculate weight index
                        const int weight_idx = (((c_in * (C_out / groups) + oc) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        const float weight_val = weight[weight_idx];
                        
                        // Calculate output contribution
                        const float contribution = in_val * weight_val;
                        
                        // Map output location to shared memory index
                        const int out_idx = ((n * C_out + oc_global) * outD + out_d) * outH * outW + out_h * outW + out_w;
                        const int shared_idx = out_idx % SHARED_MEM_SIZE;
                        
                        atomicAdd(&shared_output[shared_idx], contribution);
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Accumulate shared memory results to global memory
    for (int i = tid; i < SHARED_MEM_SIZE; i += num_threads) {
        if (shared_output[i] != 0.0f) {
            const int global_idx = (bid * SHARED_MEM_SIZE + i) % (N * C_out * outD * outH * outW);
            atomicAdd(&output[global_idx], shared_output[i]);
        }
    }
}

__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int total,
    const int C_out,
    const int outD,
    const int outH,
    const int outW) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    const int w = idx % outW; // This variable is used to calculate the output index
    int tmp = idx / outW;
    const int h = tmp % outH;
    tmp /= outH;
    const int d = tmp % outD;
    tmp /= outD;
    const int c = tmp % C_out;
    
    output[idx] += bias[c];
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
    if (bias.has_value()) CHECK_INPUT(*bias);
    
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
    const int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    const int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];
    
    const int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    const int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    const int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
    
    const int C_out = weight.size(1) * groups;
    
    auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());
    
    const int num_blocks = 1024; // Tune this based on input size
    const int shared_mem_bytes = SHARED_MEM_SIZE * sizeof(float);
    
    conv_transpose3d_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        outD, outH, outW,
        groups, C_in/groups);
    
    if (bias.has_value()) {
        const int total_elements = N * C_out * outD * outH * outW;
        const int threads = BLOCK_SIZE;
        const int blocks = (total_elements + threads - 1) / threads;
        
        add_bias_kernel<<<blocks, threads>>>(
            output.data_ptr<float>(),
            bias->data_ptr<float>(),
            total_elements, C_out, outD, outH, outW);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}