#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
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
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int index = blockIdx.x * blockDim.x + tid;
    const int total_elements = B * OC * H * W;
    
    if (index >= total_elements) return;
    
    // Decompose linear index into 4D tensor coordinates
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);
    
    float sum = 0.0f;
    
    // Process input channels in chunks to fit in shared memory
    for (int ic_chunk = 0; ic_chunk < IC; ic_chunk += BLOCK_SIZE) {
        // Cooperatively load input and weight data into shared memory
        if (ic_chunk + tid < IC) {
            const int x_offset = b * IC * H * W + (ic_chunk + tid) * H * W + h * W + w;
            const int w_offset = oc * IC + (ic_chunk + tid);
            shared_mem[tid] = x[x_offset] * weight[w_offset];
        } else {
            shared_mem[tid] = 0.0f;
        }
        __syncthreads();
        
        // Reduce within each warp using shuffle operations
        float warp_sum = shared_mem[tid];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // First thread in each warp writes result to shared memory
        if (lane == 0) {
            shared_mem[wid] = warp_sum;
        }
        __syncthreads();
        
        // Final reduction across warps
        if (tid < (BLOCK_SIZE / WARP_SIZE)) {
            sum += shared_mem[tid];
        }
        __syncthreads();
    }
    
    // First thread writes final result
    if (tid == 0) {
        output[index] = bias ? sum + bias[oc] : sum;
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

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    const int threads = BLOCK_SIZE;
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    const int shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    forward_kernel<<<blocks, threads, shared_mem_size>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}