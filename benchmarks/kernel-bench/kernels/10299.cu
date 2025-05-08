#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

define TILE_SIZE 32

declspec(device) inline float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    __shared__ float smem_x[TILE_SIZE];
    __shared__ float smem_w[TILE_SIZE];

    const int h = blockIdx.y;
    const int w = blockIdx.z;
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.w;
    
    if (oc >= OC || h >= H || w >= W || b >= B) return;
    
    float sum = 0;
    for (int ic_base = 0; ic_base < IC; ic_base += TILE_SIZE) {
        int ic = ic_base + threadIdx.y;
        float x_val = (ic < IC) ? x[b * IC * H * W + ic * H * W + h * W + w] : 0;
        float w_val = (ic < IC && oc < OC) ? weight[oc * IC + ic] : 0;
        
        smem_x[threadIdx.y] = x_val;
        smem_w[threadIdx.y] = w_val;
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE && (ic_base + i) < IC; ++i) {
            sum += smem_x[i] * smem_w[i];
        }
        __syncthreads();
    }
    
    if (oc < OC) {
        sum += bias ? bias[oc] : 0;
        output[b * OC * H * W + oc * H * W + h * W + w] = sum;
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
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    auto output = torch::empty({B, OC, H, W}, x.options());

    dim3 threads(32, TILE_SIZE);
    dim3 blocks(
        (OC + threads.x - 1) / threads.x,
        H,
        W,
        B
    );

    forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized pointwise 2D conv (CUDA)");
}
