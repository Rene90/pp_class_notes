/**
 *
 * First cuda program for adding two vectors in parallel.
 *
**/
#include <stdio.h>
#define N 1600

__global__ void vecadd(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, gridsize;
    gridsize = blockDim.x * gridDim.x;
    for(i = tid; i < N, i += gridsize) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[])
{
    int a[N], b[N], c[N];
    int *ad, *bd, *cd;

    cudaMalloc((void **) &ad, N*sizeof(int));
    cudaMalloc((void **) &bd, N*sizeof(int));
    cudaMalloc((void **) &cd, N*sizeof(int));
    read_in(a); read_in(b);
    cudaMemcpy(ad, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, N*sizeof(int), cudaMemcpyHostToDevice);

    vecadd<<<10,16>>>(ad,bd,cd);

    cudaMemcpy(c,cd,N*sizeof(int),cudaMemcpyDeviceToHost);
    write_out(c);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    return 0;
}
