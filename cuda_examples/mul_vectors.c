/**
 *
 * First cuda program for adding two vectors in parallel.
 *
**/
#include <stdio.h>

const int N = 32 * 1024;
const int threadsPerBlock = 256;
const int n_blocks = N / threadsPerBlock;

__global__ void scal_prod(float *a, float *b, float *c) {
    __shared__ float part_prod[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, size, thread_index = threadIdx.x;
    float part_res = 0;
    
    for(i = tid; i < N; i += blockDim.x * gridDim.x)
        part_res += a[i] * b[i];
    part_prod[thread_index] = part_res;
    __syncthreads();
    
    size = blockDim.x / 2;
    while(size != 0) {
        if(thread_index < size)
            part_prod[thread_index] += part_prod[thread_index + size];
        __syncthreads();
        
        size = size / 2;
    }
    if(thread_index == 0)
        c[blockIdx.x] = part_prod[0];
}

int main(int argc, char *argv[])
{
    float a[N], b[N], part_c[n_blocks];
    float *ad, *bd, *part_cd;

    cudaMalloc((void **) &ad, N*sizeof(float));
    cudaMalloc((void **) &bd, N*sizeof(float));
    cudaMalloc((void **) &part_cd, n_blocks*sizeof(float));
    read_in(a); read_in(b);
    cudaMemcpy(ad, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, N*sizeof(int), cudaMemcpyHostToDevice);

    scal_prod<<<n_blocks,threadsPerBlock>>>(ad,bd,part_cd);

    cudaMemcpy(part_c,part_cd,n_blocks*sizeof(float),cudaMemcpyDeviceToHost);
    
    c = 0;
    for(int i=0; i < n_blocks; i++) c += part_c[i];
    
    write_out(c);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(part_cd);
    return 0;
}
