#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups, int in_per_group, int c_out_per_group,
    int kernel_total, int weights_per_channel) {

    extern __shared__ float s_weights[];
    int group = blockIdx.y;

    // Load group weights to shared memory
    const float* group_base = weight + group * in_per_group * weights_per_channel;
    for (int i = threadIdx.x; i < weights_per_channel * in_per_group; i += blockDim.x) {
        s_weights[i] = group_base[i];
    }
    __syncthreads();

    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_group = N * in_per_group * D_in * H_in * W_in;
    if (elem_idx >= elements_per_group) return;

    // Decode input coordinates within group
    int w = elem_idx % W_in;
    int h = (elem_idx / W_in) % H_in;
    int d = (elem_idx / (W_in * H_in)) % D_in;
    int c_in_group = (elem_idx / (W_in * H_in * D_in)) % in_per_group;
    int n = elem_idx / (W_in * H_in * D_in * in_per_group);

    float inp = input[
        ((n * C_in + (group * in_per_group + c_in_group)) * D_in + d) * H_in * W_in + h * W_in + w
    ];

    // Base output index calculations
    int out_base_n = n * C_out * outD * outH * outW;
    int out_d_base = d * stride_d - pad_d;
    int out_h_base = h * stride_h - pad_h;
    int out_w_base = w * stride_w - pad_w;

    for (int kd = 0; kd < kernel_d; kd++) {
        int out_d = out_d_base + kd;
        if (out_d < 0 || out_d >= outD) continue;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            int out_h = out_h_base + kh;
            if (out_h < 0 || out_h >= outH) continue;
            
            for (int kw = 0; kw < kernel_w; kw++) {
                int out_w = out_w_base + kw;
                if (out_w < 0 || out_w >= outW) continue;

                for (int oc = 0; oc < c_out_per_group; oc++) {
                    int weight_offset = (c_in_group * c_out_per_group + oc) * kernel_total
                                      + ((kernel_d - 1 - kd) * kernel_h * kernel_w + (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw));
                    float w_val = s_weights[weight_offset];
                    
                    int oc_global = group * c_out_per_group + oc;
                    int out_idx = ((oc_global * outD + out_d) * outH + out_h) * outW + out_w;
                    atomicAdd(&output[out_base_n + out_idx], inp * w_val);
                }
            }
        }
    }
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

    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    int c_out_per_group = weight.size(1);
    int in_per_group = C_in / groups;
    int C_out = c_out_per_group * groups;

    int outD = (D_in - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0];
    int outH = (H_in - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1];
    int outW = (W_in - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2];

    auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());

    // Precompute values needed for kernel launch
    int kernel_total = kernel_d * kernel_h * kernel_w;
    int weights_per_channel = c_out_per_group * kernel_total;
    
    // Configure kernel grid with group dimension
    int elements_per_group = N * in_per_group * D_in * H_in * W_in;
    const int BLOCK_SIZE = 256;
    dim3 grid_dim((elements_per_group + BLOCK_SIZE - 1) / BLOCK_SIZE, groups);
    size_t shared_mem_size = in_per_group * weights_per_channel * sizeof(float);
    
    conv_transpose3d_kernel<<<grid_dim, BLOCK_SIZE, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        outD, outH, outW,
        groups, in_per_group, c_out_per_group,
        kernel_total, weights_per_channel
    );

    if (bias.has_value()) {
        auto bias_acc = bias->contiguous();
        output += bias_acc.view({1, C_out, 1, 1, 1});
    }

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D with Shared Memory Optimization");
}
