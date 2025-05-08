#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_kernel_optimized(const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ out,
                                         int64_t numel,
                                         scalar_t min_val,
                                         scalar_t max_val,
                                         unsigned int* clamp_stats) {
    // Use vectorized loads/stores when possible
    using Vec4 = typename cuda::aligned_vector<scalar_t, 4>::type;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_numel = numel / 4;
    int local_clamp_count = 0;

    // Vector processing for aligned data
    const Vec4* x_vec = reinterpret_cast<const Vec4*>(x);
    Vec4* out_vec = reinterpret_cast<Vec4*>(out);
    
    for (int i = tid; i < vec_numel; i += stride) {
        Vec4 val = x_vec[i];
        
        // Process each element of the vector
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scalar_t& elem = reinterpret_cast<scalar_t*>(&val)[j];
            if (elem < min_val) {
                elem = min_val;
                local_clamp_count++;
            } else if (elem > max_val) {
                elem = max_val;
                local_clamp_count++;
            }
        }
        
        out_vec[i] = val;
    }

    // Handle remaining elements
    for (int i = tid + vec_numel * 4; i < numel; i += stride) {
        scalar_t val = x[i];
        if (val < min_val) {
            val = min_val;
            local_clamp_count++;
        } else if (val > max_val) {
            val = max_val;
            local_clamp_count++;
        }
        out[i] = val;
    }

    // Warp-level reduction of clamp counts
    unsigned mask = 0xffffffff;
    int warp_clamps = local_clamp_count;
    
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        warp_clamps += __shfl_down_sync(mask, warp_clamps, offset);
    }

    // First thread in each warp writes to global clamp statistics
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(clamp_stats, warp_clamps);
    }
}

std::tuple<at::Tensor, int64_t> forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    // Optimal thread count based on GPU architecture
    const int threads = 256;
    const int blocks = std::min(65535, (numel + threads - 1) / threads);
    
    // Allocate memory for clamp statistics
    unsigned int* d_clamp_stats;
    cudaMalloc(&d_clamp_stats, sizeof(unsigned int));
    cudaMemset(d_clamp_stats, 0, sizeof(unsigned int));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_optimized", ([&] {
        hardtanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val),
            d_clamp_stats
        );
    }));

    // Get clamp statistics
    unsigned int h_clamp_stats;
    cudaMemcpy(&h_clamp_stats, d_clamp_stats, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_clamp_stats);

    return std::make_tuple(out, static_cast<int64_t>(h_clamp_stats));
}

std::tuple<at::Tensor, int64_t> forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized HardTanh activation with statistics (CUDA)");
}