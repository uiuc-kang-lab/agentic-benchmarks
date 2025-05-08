#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized cumulative product kernel using warp-level primitives and shared memory

template <typename scalar_t>
__global__ void cumprod_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size,
    const int stride) {

    // Each block processes one cumprod segment
    // Compute segment index from blockIdx.x
    int seg = blockIdx.x;
    int batch_idx = seg / stride;
    int in_idx = seg % stride;
    int base = batch_idx * (stride * dim_size) + in_idx;

    // Each thread loads one element if within the segment; otherwise use the multiplicative identity
    int tid = threadIdx.x;
    scalar_t val = (tid < dim_size) ? input[base + tid * stride] : static_cast<scalar_t>(1);

    // Perform warp-level inclusive scan using __shfl_up_sync with specialization for half precision
    unsigned mask = __ballot_sync(0xFFFFFFFF, tid < dim_size);
    int lane = tid & 31;
    for (int offset = 1; offset < 32; offset *= 2) {
        if constexpr (std::is_same<scalar_t, at::Half>::value) {
            __half h_val = static_cast<__half>(val);
            __half h_tmp = __shfl_up_sync(mask, h_val, offset, 32);
            if (lane >= offset && tid < dim_size) {
                h_val = __hmul(h_val, h_tmp);
            }
            val = static_cast<scalar_t>(h_val);
        } else {
            scalar_t tmp = __shfl_up_sync(mask, val, offset, 32);
            if (lane >= offset && tid < dim_size) {
                val *= tmp;
            }
        }
    }

    // Use shared memory to store the last element (i.e. the warp aggregate) of each warp
    extern __shared__ char smem[]; 
    scalar_t* warp_scan = reinterpret_cast<scalar_t*>(smem);
    int warp_id = tid / 32;
    // Determine the last valid thread index in this warp
    int warp_start = warp_id << 5;
    int warp_last = (warp_start + 31 < dim_size) ? (warp_start + 31) : (dim_size - 1);
    if (tid == warp_last) {
        warp_scan[warp_id] = val;
    }
    __syncthreads();

    // The first warp computes a sequential scan on the warp results
    int num_warps = (dim_size + 31) / 32;
    if (warp_id == 0 && tid < num_warps) {
        if (tid == 0) {
            for (int i = 1; i < num_warps; i++) {
                warp_scan[i] *= warp_scan[i - 1];
            }
        }
    }
    __syncthreads();

    // Threads in warps beyond the first multiply their per-warp result by the cumulative product from previous warps
    if (tid < dim_size && warp_id > 0) {
        scalar_t prefix = warp_scan[warp_id - 1];
        val *= prefix;
    }

    // Write the cumulative product result back to global memory
    if (tid < dim_size) {
        output[base + tid * stride] = val;
    }
}


// Host function for calling the optimized cumprod CUDA kernel

torch::Tensor cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Get tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    // Dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t total_segments = input.numel() / dim_size;

    // Set block dimension: choose the next multiple of 32 greater than or equal to dim_size
    int block_threads = ((dim_size + 31) / 32) * 32;
    dim3 block(block_threads);
    dim3 grid(total_segments);

    int num_warps = block_threads / 32;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        int shmem = num_warps * sizeof(scalar_t);
        cumprod_kernel_optimized<scalar_t><<<grid, block, shmem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward, "Optimized cumulative product forward (CUDA)");
}
