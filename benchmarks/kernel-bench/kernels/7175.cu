#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Shared memory tile size
#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 11

__device__ __forceinline__ float fetch_input(
    const float* input,
    int n, int c, int h, int w,
    int C, int H, int W) {
    if (h >= 0 && h < H && w >= 0 && w < W)
        return input[n * C * H * W + c * H * W + h * W + w];
    return 0.0f;
}

__global__ void conv2d_hybrid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OC, int outH, int outW,
    int k, int stride, int padding,
    int dilation, int groups) {
    
    __shared__ float shared_input[TILE_SIZE + MAX_KERNEL_SIZE][TILE_SIZE + MAX_KERNEL_SIZE];
    __shared__ float shared_weight[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int n = bz;
    const int m = by;
    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = blockIdx.w * TILE_SIZE + ty;
    
    const int group_id = m / (OC / groups);
    const int in_channels_per_group = C / groups;
    
    float sum = (bias != nullptr) ? bias[m] : 0.0f;
    
    // Loop over input channels
    for (int ic = group_id * in_channels_per_group; 
         ic < (group_id + 1) * in_channels_per_group; 
         ic++) {
        
        // Load input tile into shared memory
        for (int i = ty; i < TILE_SIZE + k - 1; i += blockDim.y) {
            for (int j = tx; j < TILE_SIZE + k - 1; j += blockDim.x) {
                int in_y = out_y * stride - padding + i;
                int in_x = out_x * stride - padding + j;
                shared_input[i][j] = fetch_input(input, n, ic, in_y, in_x, C, H, W);
            }
        }
        
        // Load weight tile into shared memory
        if (tx < k && ty < k) {
            shared_weight[ty][tx] = weight[m * (in_channels_per_group * k * k) + 
                                        (ic - group_id * in_channels_per_group) * (k * k) + 
                                        ty * k + tx];
        }
        
        __syncthreads();
        
        if (out_x < outW && out_y < outH) {
            // Compute convolution
            #pragma unroll
            for (int ky = 0; ky < k; ky++) {
                #pragma unroll
                for (int kx = 0; kx < k; kx++) {
                    sum += shared_input[ty * stride + ky * dilation][tx * stride + kx * dilation] * 
                          shared_weight[ky][kx];
                }
            }
        }
        
        __syncthreads();
    }
    
    if (out_x < outW && out_y < outH) {
        output[n * OC * outH * outW + 
               m * outH * outW + 
               out_y * outW + 
               out_x] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(bias.value());
    
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);
    const int k = weight.size(2);
    
    TORCH_CHECK(k <= MAX_KERNEL_SIZE, "Kernel size must be <= ", MAX_KERNEL_SIZE);
    
    const int outH = (H + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int outW = (W + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({N, OC, outH, outW}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((outW + TILE_SIZE - 1) / TILE_SIZE,
                OC,
                N,
                (outH + TILE_SIZE - 1) / TILE_SIZE);
    
    conv2d_hybrid_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, OC, outH, outW,
        k, stride, padding, dilation, groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized hybrid CUDA convolution forward");
}