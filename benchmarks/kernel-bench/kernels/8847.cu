#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fallback kernel for general stride (uses original global memory access)
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * L_out;
    if (index >= total_elements) return;

    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n = index / (L_out * C_out);

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    for (int c = 0; c < C_in; ++c) {
        for (int k = 0; k < K_w; ++k) {
            int l_in_nom = l_out + padding - k * dilation;
            if (l_in_nom % stride == 0) {
                int l_in = l_in_nom / stride;
                if (l_in >= 0 && l_in < L_in) {
                    float x_val = x[n * C_in * L_in + c * L_in + l_in];
                    float w_val = weight[c * C_out * K_w + c_out * K_w + k];
                    value += x_val * w_val;
                }
            }
        }
    }
    y[n * C_out * L_out + c_out * L_out + l_out] = value;
}

// Optimized kernel for stride==1 using shared memory tiling to align accesses
__global__ void conv_transpose1d_kernel_tiled(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int padding, int dilation)
{
    // Each block handles a contiguous tile of output positions for one (n, c_out) pair
    int tile_size = blockDim.x;  // number of output elements processed per block
    int tile_idx = blockIdx.x;   // tile index along the L_out dimension
    int nc_index = blockIdx.y;   // combined index for sample and output channel
    int n = nc_index / C_out;
    int c_out = nc_index % C_out;

    int l_out_start = tile_idx * tile_size;
    int l_thread = threadIdx.x;
    int l_out = l_out_start + l_thread;

    // Shared memory tile for x. Its width covers the output tile plus extra for the kernel span.
    extern __shared__ float shmem[];  // layout: [C_in][tile_width]
    int shmem_width = tile_size + (K_w - 1) * dilation;

    // Compute the starting global index for loading x for each channel
    // For stride==1: l_in = l_out + padding - k*dilation
    // To cover all k in [0, K_w-1] for outputs in [l_out_start, l_out_start+tile_size-1],
    // we load from: load_start = l_out_start + padding - (K_w-1)*dilation
    int load_start = l_out_start + padding - (K_w - 1) * dilation;

    // Load the required tile of x for all C_in channels into shared memory
    int total_load = C_in * shmem_width;
    for (int tid = threadIdx.x; tid < total_load; tid += blockDim.x) {
        int c = tid / shmem_width;
        int offset = tid % shmem_width;
        int l_global = load_start + offset;
        float val = 0.0f;
        if (l_global >= 0 && l_global < L_in) {
            val = x[n * C_in * L_in + c * L_in + l_global];
        }
        shmem[c * shmem_width + offset] = val;
    }
    __syncthreads();

    if (l_out < L_out) {
        float result = (bias != nullptr) ? bias[c_out] : 0.0f;
        // For each input channel and each kernel position, accumulate the contribution
        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K_w; ++k) {
                // Compute the offset within the shared memory tile
                // Derived from: x_index = (l_out + padding - k*dilation)
                // and load_start = (l_out_start + padding - (K_w-1)*dilation)
                int offset = l_thread + (K_w - 1) * dilation - k * dilation;
                if (offset >= 0 && offset < shmem_width) {
                    float x_val = shmem[c * shmem_width + offset];
                    float w_val = weight[c * C_out * K_w + c_out * K_w + k];
                    result += x_val * w_val;
                }
            }
        }
        y[n * C_out * L_out + c_out * L_out + l_out] = result;
    }
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,            // x: torch.Tensor
    py::object weight_obj,       // weight: torch.Tensor
    py::object bias_obj = py::none(),  // bias: torch.Tensor or None
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1)
{
    // Convert inputs to contiguous CUDA tensors
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

    // Extract tensor dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    
    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    if (stride == 1) {
        // Use the optimized tiled kernel when stride==1
        int tile_size = 128;  // number of output elements per block (tunable)
        dim3 block(tile_size);
        int num_tiles = (L_out + tile_size - 1) / tile_size;
        // Each block is assigned to one tile for one (n, c_out) pair
        dim3 grid(num_tiles, N * C_out);
        size_t shared_mem_size = C_in * (tile_size + (K_w - 1) * dilation) * sizeof(float);
        conv_transpose1d_kernel_tiled<<<grid, block, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, C_in, C_out, L_in, L_out, K_w,
            padding, dilation);
    } else {
        // Fallback to the original kernel for general stride
        int total_elements = N * C_out * L_out;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        conv_transpose1d_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, C_in, C_out, L_in, L_out, K_w,
            stride, padding, dilation);
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1
    );
}
