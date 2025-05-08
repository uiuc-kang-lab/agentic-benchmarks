#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute sum of squares
template <typename scalar_t>
__device__ scalar_t compute_sum_of_squares(const scalar_t* input, int batch_offset, int offset, int num_features, int numel_per_batch) {
    scalar_t sumsq = 0.0f;
    const scalar_t* ptr = input + batch_offset + offset;
    const int unroll_factor = 32;
    int n_unrolled = (num_features / unroll_factor) * unroll_factor;

    #pragma unroll
    for (int feat = 0; feat < n_unrolled; feat += unroll_factor) {
        scalar_t v0  = input[batch_offset + (feat + 0)  * numel_per_batch + offset];
        scalar_t v1  = input[batch_offset + (feat + 1)  * numel_per_batch + offset];
        scalar_t v2  = input[batch_offset + (feat + 2)  * numel_per_batch + offset];
        scalar_t v3  = input[batch_offset + (feat + 3)  * numel_per_batch + offset];
        scalar_t v4  = input[batch_offset + (feat + 4)  * numel_per_batch + offset];
        scalar_t v5  = input[batch_offset + (feat + 5)  * numel_per_batch + offset];
        scalar_t v6  = input[batch_offset + (feat + 6)  * numel_per_batch + offset];
        scalar_t v7  = input[batch_offset + (feat + 7)  * numel_per_batch + offset];
        scalar_t v8  = input[batch_offset + (feat + 8)  * numel_per_batch + offset];
        scalar_t v9  = input[batch_offset + (feat + 9)  * numel_per_batch + offset];
        scalar_t v10 = input[batch_offset + (feat + 10) * numel_per_batch + offset];
        scalar_t v11 = input[batch_offset + (feat + 11) * numel_per_batch + offset];
        scalar_t v12 = input[batch_offset + (feat + 12) * numel_per_batch + offset];
        scalar_t v13 = input[batch_offset + (feat + 13) * numel_per_batch + offset];
        scalar_t v14 = input[batch_offset + (feat + 14) * numel_per_batch + offset];
        scalar_t v15 = input[batch_offset + (feat + 15) * numel_per_batch + offset];
        scalar_t v16 = input[batch_offset + (feat + 16) * numel_per_batch + offset];
        scalar_t v17 = input[batch_offset + (feat + 17) * numel_per_batch + offset];
        scalar_t v18 = input[batch_offset + (feat + 18) * numel_per_batch + offset];
        scalar_t v19 = input[batch_offset + (feat + 19) * numel_per_batch + offset];
        scalar_t v20 = input[batch_offset + (feat + 20) * numel_per_batch + offset];
        scalar_t v21 = input[batch_offset + (feat + 21) * numel_per_batch + offset];
        scalar_t v22 = input[batch_offset + (feat + 22) * numel_per_batch + offset];
        scalar_t v23 = input[batch_offset + (feat + 23) * numel_per_batch + offset];
        scalar_t v24 = input[batch_offset + (feat + 24) * numel_per_batch + offset];
        scalar_t v25 = input[batch_offset + (feat + 25) * numel_per_batch + offset];
        scalar_t v26 = input[batch_offset + (feat + 26) * numel_per_batch + offset];
        scalar_t v27 = input[batch_offset + (feat + 27) * numel_per_batch + offset];
        scalar_t v28 = input[batch_offset + (feat + 28) * numel_per_batch + offset];
        scalar_t v29 = input[batch_offset + (feat + 29) * numel_per_batch + offset];
        scalar_t v30 = input[batch_offset + (feat + 30) * numel_per_batch + offset];
        scalar_t v31 = input[batch_offset + (feat + 31) * numel_per_batch + offset];

        sumsq += v0*v0 + v1*v1 + v2*v2 + v3*v3 +
                 v4*v4 + v5*v5 + v6*v6 + v7*v7 +
                 v8*v8 + v9*v9 + v10*v10 + v11*v11 +
                 v12*v12 + v13*v13 + v14*v14 + v15*v15 +
                 v16*v16 + v17*v17 + v18*v18 + v19*v19 +
                 v20*v20 + v21*v21 + v22*v22 + v23*v23 +
                 v24*v24 + v25*v25 + v26*v26 + v27*v27 +
                 v28*v28 + v29*v29 + v30*v30 + v31*v31;
    }

    for (int feat = n_unrolled; feat < num_features; feat++) {
        scalar_t val = input[batch_offset + feat * numel_per_batch + offset];
        sumsq += val * val;
    }

    return sumsq;
}

// Device function to normalize input
template <typename scalar_t>
__device__ void normalize_input(const scalar_t* input, scalar_t* output, int batch_offset, int offset, int num_features, int numel_per_batch, scalar_t rms) {
    const int unroll_factor = 32;
    int n_unrolled = (num_features / unroll_factor) * unroll_factor;

    #pragma unroll
    for (int feat = 0; feat < n_unrolled; feat += unroll_factor) {
        int j0  = batch_offset + (feat + 0)  * numel_per_batch + offset;
        int j1  = batch_offset + (feat + 1)  * numel_per_batch + offset;
        int j2  = batch_offset + (feat + 2)  * numel_per_batch + offset;
        int j3  = batch_offset + (feat + 3)  * numel_per_batch + offset;
        int j4  = batch_offset + (feat + 4)  * numel_per_batch + offset;
        int j5  = batch_offset + (feat + 5)  * numel_per_batch + offset;
        int j6  = batch_offset + (feat + 6)  * numel_per_batch + offset;
        int j7  = batch_offset + (feat + 7)  * numel_per_batch + offset;
        int j8  = batch_offset + (feat + 8)  * numel_per_batch + offset;
        int j9  = batch_offset + (feat + 9)  * numel_per_batch + offset;
        int j10 = batch_offset + (feat + 10) * numel_per_batch + offset;
        int j11 = batch_offset + (feat + 11) * numel_per_batch + offset;
        int j12 = batch_offset + (feat + 12) * numel_per_batch + offset;
        int j13 = batch_offset + (feat + 13) * numel_per_batch + offset;
        int j14 = batch_offset + (feat + 14) * numel_per_batch + offset;
        int j15 = batch_offset + (feat + 15) * numel_per_batch + offset;
        int j16 = batch_offset + (feat + 16) * numel_per_batch + offset;
        int j17 = batch_offset + (feat + 17) * numel_per_batch + offset;
        int j18 = batch_offset + (feat + 18) * numel_per_batch + offset;
        int j19 = batch_offset + (feat + 19) * numel_per_batch + offset;
        int j20 = batch_offset + (feat + 20) * numel_per_batch + offset;
        int j21 = batch_offset + (feat + 21) * numel_per_batch + offset;
        int j22 = batch_offset + (feat + 22) * numel_per_batch + offset;
        int j23 = batch_offset + (feat + 23) * numel_per_batch + offset;
        int j24 = batch_offset + (feat + 24) * numel_per_batch + offset;
        int j25 = batch_offset + (feat + 25) * numel_per_batch + offset;
        int j26 = batch_offset + (feat + 26) * numel_per_batch + offset;
        int j27 = batch_offset + (feat + 27) * numel_per_batch + offset;
        int j28 = batch_offset + (feat + 28) * numel_per_batch + offset;
        int j29 = batch_offset + (feat + 29) * numel_per_batch + offset;
        int j30 = batch_offset + (feat + 30) * numel_per_batch + offset;
        int j31 = batch_offset + (feat + 31) * numel_per_batch + offset;

        output[j0]  = input[j0]  / rms;
        output[j1]  = input[j1]  / rms;
        output[j2]  = input[j2]  / rms;
        output[j3]  = input[j3]  / rms;
        output[j4]  = input[j4]  / rms;
        output[j5]  = input[j5]  / rms;
        output[j6]  = input[j6]  / rms;
        output[j7]  = input[j7]  / rms;
        output[j8]  = input[j8]  / rms;
        output[j9]  = input[j9]  / rms;
        output[j10] = input[j10] / rms;
        output[j11] = input[j11] / rms;
        output[j12] = input[j12] / rms;
        output[j13] = input[j13] / rms;
        output[j14] = input[j14] / rms;
        output[j15] = input[j15] / rms;
        output[j16] = input[j16] / rms;
        output[j17] = input[j17] / rms;
        output[j18] = input[j18] / rms;
        output[j19] = input[j19] / rms;
        output[j20] = input[j20] / rms;
        output[j21] = input[j21] / rms;
        output[j22] = input[j22] / rms;
        output[j23] = input[j23] / rms;
        output[j24] = input[j24] / rms;
        output[j25] = input[j25] / rms;
        output[j26] = input[j26] / rms;
        output[j27] = input[j27] / rms;
        output[j28] = input[j28] / rms;
        output[j29] = input[j29] / rms;
        output[j30] = input[j30] / rms;
        output[j31] = input[j31] / rms;
    }

    for (int feat = n_unrolled; feat < num_features; feat++) {
        int j = batch_offset + feat * numel_per_batch + offset;
        output[j] = input[j] / rms;
    }
}

// Kernel function
template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int total_samples = batch_size * numel_per_batch;
    const int stride = blockDim.x * gridDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_samples; idx += stride) {
        int batch_id = idx / numel_per_batch;
        int offset = idx % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;

        // Compute sum of squares
        scalar_t sumsq = compute_sum_of_squares(input, batch_offset, offset, num_features, numel_per_batch);

        // Compute RMS
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize input
        normalize_input(input, output, batch_offset, offset, num_features, numel_per_batch, rms);
    }
}

// Host function
torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_samples = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int max_blocks = 65535;
    int blocks = (total_samples + threads_per_block - 1) / threads_per_block;
    if (blocks > max_blocks) blocks = max_blocks;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "Modular RMS normalization forward (CUDA)");
}
