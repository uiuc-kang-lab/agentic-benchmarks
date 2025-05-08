#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using shared memory and warp-level primitives for intra-block reduction
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
    // Each block handles one output element in the (B, OC, H, W) tensor
    int out_idx = blockIdx.x;
    int total_outputs = B * OC * H * W;
    if (out_idx >= total_outputs) return;

    // Decompose linear output index into 4D coordinates
    int w = out_idx % W;
    int h = (out_idx / W) % H;
    int oc = (out_idx / (W * H)) % OC;
    int b = out_idx / (W * H * OC);

    float partial = 0.0f;
    // Each thread in the block computes a partial sum over the input channels
    for (int ic = threadIdx.x; ic < IC; ic += blockDim.x) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        int weight_offset = oc * IC + ic; // weight shape is [OC, IC, 1, 1] flattened as [OC, IC]
        partial += x[x_offset] * weight[weight_offset];
    }

    // Use warp-level reduction to reduce the partial sums within each warp
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }

    // Allocate shared memory to hold the reduced sums from each warp
    __shared__ float warpSums[32]; // 32 is enough since maximum warps per block <= blockDim.x/32
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    // First thread in each warp writes the result
    if (lane == 0) {
        warpSums[warpId] = partial;
    }
    __syncthreads();

    // Final reduction: first warp reduces all warp sums
    if (warpId == 0) {
        // Load this thread's value from shared memory if it's within bounds
        float sum = (lane < ((blockDim.x + 31) >> 5)) ? warpSums[lane] : 0.0f;
        
        // Perform warp-wide reduction
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        // First thread writes the final result
        if (lane == 0) {
            if (bias != nullptr) {
                sum += bias[oc];
            }
            output[out_idx] = sum;
        }
    }
}

// CUDA interface function
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

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) {
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    // Create the output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // One block per output element
    const int total_outputs = B * OC * H * W;
    const int threads = 256;
    dim3 blocks(total_outputs);

    forward_kernel<<<blocks, threads>>>(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized Pointwise 2D convolution forward (CUDA)");
}
