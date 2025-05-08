#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void hybrid_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    if (num_classes <= 32) {
        for (int i = gid; i < batch_size; i += stride) {
            const float* logits_i = logits + i * num_classes;
            int target = targets[i];
            
            float max_logit = -FLT_MAX;
            float sum_exp = 0.0f;
            
            for (int j = 0; j < num_classes; j++) {
                float val = logits_i[j];
                max_logit = fmaxf(max_logit, val);
            }
            
            for (int j = 0; j < num_classes; j++) {
                sum_exp += expf(logits_i[j] - max_logit);
            }
            
            losses[i] = -(logits_i[target] - max_logit - logf(sum_exp));
        }
    } else {
        float* s_max = sdata;
        float* s_sum = sdata + blockDim.x;
        
        int samples_per_block = min(32, (batch_size - bid * 32) + 1);
        for (int sample_offset = 0; sample_offset < samples_per_block; sample_offset++) {
            int i = bid * 32 + sample_offset;
            if (i >= batch_size) break;
            
            float local_max = -FLT_MAX;
            for (int j = tid; j < num_classes; j += blockDim.x) {
                float val = logits[i * num_classes + j];
                local_max = fmaxf(local_max, val);
            }
            s_max[tid] = local_max;
            __syncthreads();
            
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
                }
                __syncthreads();
            }
            float max_val = s_max[0];
            
            float local_sum = 0.0f;
            for (int j = tid; j < num_classes; j += blockDim.x) {
                local_sum += expf(logits[i * num_classes + j] - max_val);
            }
            s_sum[tid] = local_sum;
            __syncthreads();
            
            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_sum[tid] += s_sum[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                int target = targets[i];
                float target_logit = logits[i * num_classes + target];
                losses[i] = -(target_logit - max_val - logf(s_sum[0]));
            }
        }
    }
}