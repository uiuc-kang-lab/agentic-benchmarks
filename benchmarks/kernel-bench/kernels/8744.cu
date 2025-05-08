#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void conv_transpose1d_kernel_warp_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    extern __shared__ float partial_sums[];
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    int index = blockIdx.x * blockDim.x + tid;
    int total_elements = N * C_out * L_out;
    
    if (index < total_elements) {
        const int l_out = index % L_out;
        const int c_out = (index / L_out) % C_out;
        const int n = index / (L_out * C_out);

        float sum = 0.0f;
        
        // Each thread processes its portion of input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int x_batch_offset = n * C_in * L_in + c_in * L_in;
            const int w_offset = c_in * C_out * K_w + c_out * K_w;
            
            #pragma unroll 4
            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int l_in_nom = l_out + padding - k_w * dilation;
                const int mod_valid = (l_in_nom % stride == 0);
                const int l_in = l_in_nom / stride;
                const int range_valid = (l_in >= 0 && l_in < L_in);
                
                if (mod_valid && range_valid) {
                    sum += x[x_batch_offset + l_in] * weight[w_offset + k_w];
                }
            }
        }

        // Warp-level reduction
        sum = warpReduceSum(sum);

        // First thread in each warp writes to shared memory
        if (lane == 0) {
            partial_sums[wid] = sum;
        }
    }
    
    __syncthreads();

    // Final reduction across warps using first warp
    if (wid == 0 && index < total_elements) {
        float final_sum = (lane < warps_per_block) ? partial_sums[lane] : 0.0f;
        final_sum = warpReduceSum(final_sum);

        if (lane == 0) {
            const int l_out = index % L_out;
            const int c_out = (index / L_out) % C_out;
            const int n = index / (L_out * C_out);
            
            if (bias != nullptr) {
                final_sum += bias[c_out];
            }
            
            y[n * C_out * L_out + c_out * L_out + l_out] = final_sum;
        }
    }
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    const int threads = 256;
    const int warps_per_block = threads / WARP_SIZE;
    const int blocks = (N * C_out * L_out + threads - 1) / threads;
    const int shared_mem_size = warps_per_block * sizeof(float);

    conv_transpose1d_kernel_warp_optimized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Conv Transpose1D forward with warp-level optimization (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}