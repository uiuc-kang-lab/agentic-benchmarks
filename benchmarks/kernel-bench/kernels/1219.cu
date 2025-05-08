#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void einsum_kernel_streamed_unrolled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH_chunk, int I, int J, int L, int K
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BATCH_chunk * I * J * K;
    if (global_idx >= total) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b_local = remainder / I;

    float sum = 0.0f;
    
    // Manual loop unrolling for L dimension
    int l = 0;
    #pragma unroll 4
    for(; l + 3 < L; l += 4) {
        int a_offset0 = b_local * I*J*L + i*J*L + j*L + l;
        int a_offset1 = a_offset0 + 1;
        int a_offset2 = a_offset0 + 2;
        int a_offset3 = a_offset0 + 3;
        
        int b_offset0 = l*K + k;
        int b_offset1 = (l+1)*K + k;
        int b_offset2 = (l+2)*K + k;
        int b_offset3 = (l+3)*K + k;
        
        sum += A[a_offset0] * B[b_offset0] +
               A[a_offset1] * B[b_offset1] +
               A[a_offset2] * B[b_offset2] +
               A[a_offset3] * B[b_offset3];
    }
    
    // Handle remaining elements
    for(; l < L; l++) {
        int a_offset = b_local * I*J*L + i*J*L + j*L + l;
        int b_offset = l*K + k;
        sum += A[a_offset] * B[b_offset];
    }
    
    C[global_idx] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    int num_streams = 4;
    int batch_chunk = (BATCH + num_streams - 1) / num_streams;

    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s]);
    }

    for (int s = 0; s < num_streams; s++) {
        int start_batch = s * batch_chunk;
        int current_batch = std::min(batch_chunk, BATCH - start_batch);
        if (current_batch <= 0) break;

        const float* A_ptr = A.data_ptr<float>() + start_batch * I * J * L;
        float* C_ptr = C.data_ptr<float>() + start_batch * I * J * K;
        int total_elements = current_batch * I * J * K;

        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        einsum_kernel_streamed_unrolled<<<blocks, threads, 0, streams[s]>>>(
            A_ptr, B.data_ptr<float>(), C_ptr,
            current_batch, I, J, L, K
        );
    }

    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}