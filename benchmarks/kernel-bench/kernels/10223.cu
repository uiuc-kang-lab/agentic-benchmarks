#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the number of threads used per output element (should be a multiple of warp size)
#define THREADS_PER_OUTPUT 64

// Kernel: each group of THREADS_PER_OUTPUT threads cooperatively computes one output element
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
    // Total number of output elements
    int total_output = B * OC * H * W;

    // Each block is organized so that groups of THREADS_PER_OUTPUT threads compute one output element
    int group_id = threadIdx.x / THREADS_PER_OUTPUT;            // which output element within the block tile
    int thread_in_group = threadIdx.x % THREADS_PER_OUTPUT;       // thread index within the group
    int groups_per_block = blockDim.x / THREADS_PER_OUTPUT;         

    // Determine the global output element index
    int output_idx = blockIdx.x * groups_per_block + group_id;
    if (output_idx >= total_output) return;

    // Decode the global output index into 4D tensor coordinates: (b, oc, h, w)
    int w_out = output_idx % W;
    int h_out = (output_idx / W) % H;
    int oc = (output_idx / (W * H)) % OC;
    int b = output_idx / (W * H * OC);

    // Each thread computes a partial sum over the input channels in a strided manner
    float local_sum = 0.0f;
    for (int ic = thread_in_group; ic < IC; ic += THREADS_PER_OUTPUT) {
        int x_offset = b * IC * H * W + ic * H * W + h_out * W + w_out;
        int w_offset = oc * IC + ic;
        local_sum += x[x_offset] * weight[w_offset];
    }

    // Use warp-level shuffle reductions within each warp
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Now, each warp (of 32 threads) has its lane 0 holding the partial sum for that warp.
    // Since THREADS_PER_OUTPUT is 64, there are 2 warps per output element. 
    // Use shared memory to combine the two warp results.
    extern __shared__ float shared_data[]; // shared memory size is (groups_per_block * 2) * sizeof(float)
    if ((thread_in_group & 31) == 0) {  // thread is lane 0 within its warp
        int warp_id = thread_in_group >> 5; // 0 for first warp, 1 for second warp
        int shared_index = group_id * 2 + warp_id; 
        shared_data[shared_index] = local_sum;
    }
    __syncthreads();

    // Final reduction: let the first thread of the group combine the two warp results
    if (thread_in_group == 0) {
        int base_index = group_id * 2;
        float sum_final = shared_data[base_index] + shared_data[base_index + 1];
        if (bias != nullptr) {
            sum_final += bias[oc];
        }
        output[output_idx] = sum_final;
    }
}


// CUDA forward function exposed to PyTorch
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
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

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());
    
    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* o_ptr = output.data_ptr<float>();

    // Determine grid and block dimensions
    const int threads = 256; // Total threads per block
    // Each group of THREADS_PER_OUTPUT threads computes one output element
    int groups_per_block = threads / THREADS_PER_OUTPUT; // e.g., 256/64 = 4 output elements per block
    int total_output = B * OC * H * W;
    int blocks = (total_output + groups_per_block - 1) / groups_per_block;

    // Dynamic shared memory size: each group uses 2 floats
    size_t shared_mem_bytes = groups_per_block * 2 * sizeof(float);

    // Launch the kernel
    forward_kernel<<<blocks, threads, shared_mem_bytes>>>(
        x_ptr, w_ptr, b_ptr, o_ptr, B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized reduction pointwise 2D convolution forward (CUDA)");
}
