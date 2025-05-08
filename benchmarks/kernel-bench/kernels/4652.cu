#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel: uses grid-stride loop over samples and manual unrolling for feature reduction
// This kernel processes each sample (a specific offset in the batch) by accumulating the sum of squares
// with 32-wide unrolling and then normalizes using the computed RMS value.

template <typename scalar_t>
__global__ void rms_norm_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Each thread processes one sample across features; total_samples = batch_size * numel_per_batch
    const int total_samples = batch_size * numel_per_batch;
    const int stride = blockDim.x * gridDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_samples; idx += stride) {
        int batch_id = idx / numel_per_batch;
        int offset = idx % numel_per_batch;
        int batch_offset = batch_id * num_features * numel_per_batch;

        // Compute sum of squares with manual unrolling
        scalar_t sumsq = 0;
        const int unroll_factor = 32;
        int n_unrolled = (num_features / unroll_factor) * unroll_factor;

        #pragma unroll
        for (int feat = 0; feat < n_unrolled; feat += unroll_factor) {
            scalar_t v0  = __ldg(&input[batch_offset + (feat + 0)  * numel_per_batch + offset];
            scalar_t v1  = __ldg(&input[batch_offset + (feat + 1)  * numel_per_batch + offset];
            scalar_t v2  = __ldg(&input[batch_offset + (feat + 2)  * numel_per_batch + offset];
            scalar_t v3  = __ldg(&input[batch_offset + (feat + 3)  * numel_per_batch + offset];
            scalar_t v4  = __ldg(&input[batch_offset + (feat + 4)  * numel_per_batch + offset];
            scalar_t v5  = __ldg(&input[batch_offset + (feat + 5)  * numel_per_batch + offset];
            scalar_t v6  = __ldg(&input[batch_offset + (feat + 6)  * numel_per_batch + offset];
            scalar_t v7  = __ldg(&input[batch_offset + (feat + 7)  * numel_per_batch + offset];
            scalar_t v8  = __ldg(&input[batch_offset + (feat + 8)  * numel_per_batch + offset];
            scalar_t v9  = __ldg(&input[batch_offset + (feat + 9)  * numel_per_batch + offset];
            scalar_t v10 = __ldg(&input[batch_offset + (feat + 10) * numel_per_batch + offset];
            scalar_t v11 = __ldg(&input[batch_offset + (feat + 11) * numel_per_batch + offset];
            scalar_t v12 = __ldg(&input[batch_offset + (feat + 12) * numel_per_batch + offset];
            scalar_t v13 = __ldg(&input[batch_offset + (feat + 13) * numel_per_batch + offset];
            scalar_t v14 = __ldg(&input[batch_offset + (feat + 14) * numel_per_batch + offset];
            scalar_t v15 = __ldg(&input[batch_offset + (feat + 15) * numel_per_batch + offset];
            scalar_t v16 = __ldg(&input[batch_offset + (feat + 16) * numel_per_batch + offset];
            scalar_t v17 = __ldg(&input[batch_offset + (feat + 17) * numel_per_batch + offset];
            scalar_t v18 = __ldg(&input[batch_offset + (feat + 18) * numel_per_batch + offset];
            scalar_t v19 = __ldg(&input[batch_offset + (feat + 19) * numel_per_batch + offset];
            scalar_t v20 = __ldg(&input[batch_offset + (feat + 20) * numel_per_batch + offset];
            scalar_t v21 = __ldg(&input[batch_offset + (feat + 21) * numel_per_batch + offset];
            scalar_t v22 = __ldg(&input[batch_offset + (feat + 22) * numel_per_batch + offset];
            scalar_t v23 = __ldg(&input[batch_offset + (feat + 23) * numel_per_batch + offset];
            scalar_t v24 = __ldg(&input[batch_offset + (feat + 24) * numel_per_batch + offset];
            scalar_t v25 = __ldg(&input[batch_offset + (feat + 25) * numel_per_batch + offset];
            scalar_t v26 = __ldg(&input[batch_offset + (feat + 26) * numel_per_batch + offset];
            scalar_t v27 = __ldg(&input[batch_offset + (feat + 27) * numel_per_batch + offset];
            scalar_t v28 = __ldg(&input[batch_offset + (feat + 28) * numel_per_batch + offset];
            scalar_t v29 = __ldg(&input[batch_offset + (feat + 29) * numel_per_batch + offset];
            scalar_t v30 = __ldg(&input[batch_offset + (feat + 30) * numel_per_batch + offset];
            scalar_t v31 = __ldg(&input[batch_offset + (feat + 31) * numel_per_batch + offset];

            sumsq += v0*v0 + v1*v1 + v2*v2 + v3*v3 +
                     v4*v4 + v5*v5 + v6*v6 + v7*v7 +
                     v8*v8 + v9*v9 + v10*v10 + v11*v11 +
                     v12*v12 + v13*v13 + v14*v14 + v15*v15 +
                     v16*v16 + v17*v17 + v18*v18 + v19*v19 +
                     v20*v20 + v21*v21 + v22*v22 + v23*v23 +
                     v24*v24 + v25*v25 + v26*v26 + v27*v27 +
                     v28*v28 + v29*v29 + v30*v30 + v31*v31;
        }

        // Process remaining features
        for (int feat = n_unrolled; feat < num_features; feat++) {
            scalar_t val = __ldg(&input[batch_offset + feat * numel_per_batch + offset];
            sumsq += val * val;
        }

        // Compute RMS
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize input values with the computed RMS (using similar unrolling)
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

            output[j0]  = __ldg(&input[j0]  / rms;
            output[j1]  = __ldg(&input[j1]  / rms;
            output[j2]  = __ldg(&input[j2]  / rms;
            output[j3]  = __ldg(&input[j3]  / rms;
            output[j4]  = __ldg(&input[j4]  / rms;
            output[j5]  = __ldg(&input[j5]  / rms;
            output[j6]  = __ldg(&input[j6]  / rms;
            output[j7]  = __ldg(&input[j7]  / rms;
            output[j8]  = __ldg(&input[j8]  / rms;
            output[j9]  = __ldg(&input[j9]  / rms;
            output[j10] = __ldg(&input[j10] / rms;
            output[j11] = __ldg(&input[j11] / rms;
            output[j12] = __ldg(&input[j12] / rms;
            output[j13] = __ldg(&input[j13] / rms;
            output[j14] = __ldg(&input[j14] / rms;
            output[j15] = __ldg(&input[j15] / rms;
            output[j16] = __ldg(&input[j16] / rms;
            output[j17] = __ldg(&input[j17] / rms;
            output[j18] = __ldg(&input[j18] / rms;
            output[j19] = __ldg(&input[j19] / rms;
            output[j20] = __ldg(&input[j20] / rms;
            output[j21] = __ldg(&input[j21] / rms;
            output[j22] = __ldg(&input[j22] / rms;
            output[j23] = __ldg(&input[j23] / rms;
            output[j24] = __ldg(&input[j24] / rms;
            output[j25] = __ldg(&input[j25] / rms;
            output[j26] = __ldg(&input[j26] / rms;
            output[j27] = __ldg(&input[j27] / rms;
            output[j28] = __ldg(&input[j28] / rms;
            output[j29] = __ldg(&input[j29] / rms;
            output[j30] = __ldg(&input[j30] / rms;
            output[j31] = __ldg(&input[j31] / rms;
        }
        
        // Process any trailing features
        for (int feat = n_unrolled; feat < num_features; feat++) {
            int j = batch_offset + feat * numel_per_batch + offset;
            output[j] = __ldg(&input[j] / rms;
        }
    }
}

// Host function that launches the kernel

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
        rms_norm_kernel_combined<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "Combined RMS normalization forward (CUDA)");
}
