#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void streamed_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M, int64_t K, int64_t start_row) {
    
    int row = start_row + blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    __shared__ scalar_t warp_results[32];
    scalar_t thread_sum = 0;

    const scalar_t* A_row = A[row].data();
    const scalar_t* B_ptr = B.data();

    if constexpr (sizeof(scalar_t) == 4) {
        using vec_t = float4;
        int num_vec = K / 4;
        const vec_t* A_vec = reinterpret_cast<const vec_t*>(A_row);
        const vec_t* B_vec = reinterpret_cast<const vec_t*>(B_ptr);

        for (int i = tid; i < num_vec; i += blockDim.x) {
            vec_t a = __ldg(&A_vec[i]);
            vec_t b = __ldg(&B_vec[i]);
            thread_sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }

        int rem = num_vec * 4;
        for (int i = rem + tid; i < K; i += blockDim.x) {
            thread_sum += __ldg(&A_row[i]) * __ldg(&B_ptr[i]);
        }
    } else {
        for (int i = tid; i < K; i += blockDim.x) {
            thread_sum += __ldg(&A_row[i]) * __ldg(&B_ptr[i]);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    if (lane == 0)
        warp_results[wid] = thread_sum;
    __syncthreads();

    if (wid == 0 && lane < (blockDim.x >> 5)) {
        scalar_t final = warp_results[lane];
        for (int offset = 16; offset > 0; offset >>= 1)
            final += __shfl_down_sync(0xffffffff, final, offset);
        if (lane == 0)
            C[row][0] = final;
    }
}

torch::Tensor streamed_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    int64_t M = A.size(0), K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    
    auto C = torch::empty({M, 1}, A.options());
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
        cudaStreamCreate(&streams[i]);

    const int64_t chunk = (M + num_streams - 1) / num_streams;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "streamed_matvec", [&] {
        auto B_flat = B.view({-1});
        for (int i = 0; i < num_streams; ++i) {
            int64_t start = i * chunk;
            int64_t end = std::min(start + chunk, M);
            if (start >= end) continue;
            
            streamed_matvec_kernel<scalar_t><<<end-start, threads, 0, streams[i]>>>(
                A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                M, K, start);
        }
    });

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_matvec_mul_cuda, "Streamed Matrix-Vector Multiplication (CUDA)");
}