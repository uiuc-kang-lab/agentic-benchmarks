#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for warp-level inclusive scan using shuffle intrinsics
__device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n_val;
        }
    }
    return val;
}

// Device function to compute reverse cumulative sum for a single row
__device__ void compute_reverse_cumsum(const float* input, float* output, int64_t n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    float val = 0;
    if (tid < n) {
        val = input[n - 1 - tid];
    }

    val = warp_inclusive_scan(val);

    __shared__ float warp_sums[32];
    if (lane == 31 || tid == n - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (tid < n) {
        float offset = 0;
        for (int i = 0; i < warp_id; ++i) {
            offset += warp_sums[i];
        }
        output[n - 1 - tid] = val + offset;
    }
}

// Kernel to compute reverse cumulative sum along the last dimension
__global__ void reverse_cumsum_kernel(const float* input, float* output, int64_t n, int64_t outer) {
    int row = blockIdx.x;
    if (row < outer) {
        compute_reverse_cumsum(input + row * n, output + row * n, n);
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
        while (threads < n) {
            threads *= 2;
        }
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadBlock(threads);

        reverse_cumsum_kernel<<<blocks, threadBlock>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            n, outer);
    } else {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Modular reverse cumulative sum (CUDA)");
}