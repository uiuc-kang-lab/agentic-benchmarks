#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define CHANNEL_TILES 32

__global__ void forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    __shared__ float shared_x[CHANNEL_TILES][BLOCK_SIZE + 1];
    __shared__ float shared_w[CHANNEL_TILES][BLOCK_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int h = by * BLOCK_SIZE + ty;
    const int w = bx * BLOCK_SIZE + tx;
    
    if (h >= H || w >= W) return;
    
    float sum = 0.0f;
    
    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < OC; ++oc) {
            // Process input channels in tiles
            for (int ic_start = 0; ic_start < IC; ic_start += CHANNEL_TILES) {
                const int ic_end = min(ic_start + CHANNEL_TILES, IC);
                
                // Cooperatively load input data into shared memory
                if (tx < (ic_end - ic_start)) {
                    shared_x[tx][ty] = x[b * IC * H * W + (ic_start + tx) * H * W + h * W + w];
                    shared_w[tx][ty] = weight[oc * IC + ic_start + tx];
                }
                __syncthreads();
                
                // Compute partial sum for this tile
                #pragma unroll
                for (int i = 0; i < (ic_end - ic_start); ++i) {
                    sum += shared_x[i][ty] * shared_w[i][ty];
                }
                __syncthreads();
            }
            
            // Write output with bias
            if (bias) {
                sum += bias[oc];
            }
            output[b * OC * H * W + oc * H * W + h * W + w] = sum;
            sum = 0.0f;
        }
    }
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    
    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);
    
    auto output = torch::empty({B, OC, H, W}, x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((W + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, IC, OC, H, W
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}