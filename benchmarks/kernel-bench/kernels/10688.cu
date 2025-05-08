#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void aligned_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                            scalar_t* __restrict__ output,
                                            const int64_t n) {
    int row = blockIdx.x;
    const int64_t row_offset = row * n;
    const int tid = threadIdx.x;
    
    // Ensure aligned access by having threads cooperate on loading consecutive elements
    __shared__ scalar_t shared_data[1024 + 32];  // Extra space for padding
    
    // Load data in aligned chunks of 4 elements when possible
    const int aligned_n = (n + 3) / 4 * 4;  // Round up to multiple of 4
    const scalar_t* aligned_input = input + row_offset;
    
    #pragma unroll
    for (int i = tid * 4; i < aligned_n; i += blockDim.x * 4) {
        if (i < n) {
            // Use __ldg for read-only global memory access
            float4 chunk;
            if (i + 3 < n) {
                // Load full chunk of 4 elements
                const float4* src = reinterpret_cast<const float4*>(&aligned_input[n - 4 - i]);
                chunk = *reinterpret_cast<const float4*>(__ldg(reinterpret_cast<const float*>(src)));
            } else {
                // Handle boundary case
                int remaining = n - i;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (j < remaining) {
                        chunk.x = __ldg(&aligned_input[n - 1 - (i + j)]);
                    }
                }
            }
            
            // Store to shared memory in reverse order
            if (i + 3 < n) {
                shared_data[i] = chunk.w;
                shared_data[i + 1] = chunk.z;
                shared_data[i + 2] = chunk.y;
                shared_data[i + 3] = chunk.x;
            } else {
                int remaining = n - i;
                #pragma unroll
                for (int j = 0; j < remaining; j++) {
                    shared_data[i + j] = (&chunk.x)[3 - j];
                }
            }
        }
    }
    __syncthreads();

    // Compute prefix sum within each warp
    scalar_t sum = 0;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t n_val = __shfl_up_sync(0xffffffff, sum, offset);
        if (lane >= offset) {
            sum += n_val;
        }
    }

    // Store warp sums and compute warp offsets
    __shared__ scalar_t warp_sums[32];
    __shared__ scalar_t warp_offsets[32];
    
    if ((lane == 31) || (tid == n - 1)) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        scalar_t running_sum = 0;
        #pragma unroll
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            warp_offsets[i] = running_sum;
            running_sum += warp_sums[i];
        }
    }
    __syncthreads();

    // Add warp offsets and write final results
    if (warp_id > 0) {
        sum += warp_offsets[warp_id];
    }

    // Write results back to global memory in aligned chunks when possible
    #pragma unroll
    for (int i = tid * 4; i < n; i += blockDim.x * 4) {
        float4 output_chunk;
        int remaining = min(4, n - i);
        
        #pragma unroll
        for (int j = 0; j < remaining; j++) {
            (&output_chunk.x)[j] = shared_data[i + j];
        }
        
        if (remaining == 4) {
            // Write aligned chunk
            *reinterpret_cast<float4*>(&output[row_offset + i]) = output_chunk;
        } else {
            // Handle boundary case
            #pragma unroll
            for (int j = 0; j < remaining; j++) {
                output[row_offset + i + j] = (&output_chunk.x)[j];
            }
        }
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
        // Round up threads to multiple of warp size for better memory alignment
        int threads = ((n + 31) / 32) * 32;
        if (threads > 1024) threads = 1024;

        dim3 blocks(outer);
        dim3 threadBlock(threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "aligned_reverse_cumsum_cuda", ([&] {
            aligned_reverse_cumsum_kernel<scalar_t><<<blocks, threadBlock>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n);
        }));
    } else {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Aligned reverse cumulative sum (CUDA)");
}