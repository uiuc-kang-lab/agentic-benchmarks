#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel does a reverse cumulative sum along the last dimension
// but aims to reduce warp divergence by unifying the conditional logic.

// The kernel assumes that the last dimension is contiguous and that n <= 1024.

// Device function for warp-level inclusive scan
__device__ float warp_inclusive_scan(float val, int lane) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }
    return val;
}

// Kernel for reverse cumulative sum with unified control flow to avoid warp divergence
__global__ void uniform_reverse_cumsum_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              const int64_t n) {
    // Each block processes one row
    int row = blockIdx.x;
    const int64_t row_offset = row * n;
    int tid = threadIdx.x;
    int lane = tid & 31;       // lane index within a warp
    int warp_id = tid >> 5;    // warp index within the block

    __shared__ float warp_sums[32];
    __shared__ float shared_offset;
    __shared__ int warp_done[32];  // Track how many warps are done
    if (tid == 0) shared_offset = 0;
    __syncthreads();

    // All threads execute this if-condition uniformly, mitigating divergence effects.
    float val = (tid < n) ? input[row_offset + (n - 1 - tid)] : 0;
    val = warp_inclusive_scan(val, lane);

    if ((lane == 31 || tid == n - 1) && tid < n) {
        float warp_sum = val;
        atomicAdd(&shared_offset, warp_sum);
        warp_done[warp_id] = 1;  // Mark this warp as done
    }
    __syncthreads();

    // Broadcast offset to all threads once all warps are done
    if (tid == 0) {
        // Ensure all warps have finished their sums before setting shared_offset
        int all_done = 0;
        while (!all_done) {
            all_done = 1;
            for (int i = 0; i < 32; i++) {
                if (!warp_done[i]) {
                    all_done = 0;
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (tid < n) {
        output[row_offset + (n - 1 - tid)] = val + shared_offset;
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    if (dim == ndim - 1 && n <= 1024) {
        int threads = 1;
        while (threads < n) threads *= 2;
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadBlock(threads);

        uniform_reverse_cumsum_kernel<<<blocks, threadBlock>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), n);
    } else {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum with uniform control flow (CUDA)");
}
