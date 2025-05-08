#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

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
    __shared__ float x_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float w_shared[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    
    const int bx = blockIdx.x % ((W + TILE_SIZE - 1) / TILE_SIZE);
    const int by = blockIdx.y % ((H + TILE_SIZE - 1) / TILE_SIZE);
    const int bz = blockIdx.z;
    
    const int batch_idx = bz / OC;
    const int oc = bz % OC;
    
    const int h_out = by * TILE_SIZE + ty;
    const int w_out = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over input channel tiles
    for (int ic_tile = 0; ic_tile < IC; ic_tile += TILE_SIZE) {
        // Collaborative loading of input tile
        if (h_out < H && w_out < W && (ic_tile + tx) < IC) {
            x_shared[ty][tx] = x[batch_idx * IC * H * W +
                                (ic_tile + tx) * H * W +
                                h_out * W + w_out];
        } else {
            x_shared[ty][tx] = 0.0f;
        }
        
        // Collaborative loading of weight tile
        if (ty < TILE_SIZE && (ic_tile + tx) < IC) {
            w_shared[ty][tx] = weight[oc * IC + ic_tile + tx];
        } else {
            w_shared[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((ic_tile + k) < IC) {
                sum += x_shared[ty][k] * w_shared[0][k];
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (h_out < H && w_out < W) {
        const int out_idx = batch_idx * OC * H * W +
                           oc * H * W +
                           h_out * W + w_out;
        output[out_idx] = bias ? sum + bias[oc] : sum;
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
    
    const dim3 threads(TILE_SIZE * TILE_SIZE);
    const dim3 blocks(
        (W + TILE_SIZE - 1) / TILE_SIZE,
        (H + TILE_SIZE - 1) / TILE_SIZE,
        B * OC
    );
    
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