#include <stdio.h>

#define NBLOCKS  2
#define TPERBLK  1

#define NTHREADS (NBLOCKS * TPERBLK)

void errCheck()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error %d: %s\n", err, cudaGetErrorString(err));
        exit(1);
    }
}


// @@@code  {inter block atomic race}   {cuda:bbatomic}
__device__ int lock = 0;
__device__ int dummy = 0;

__global__ void kmain(unsigned int *data)   // @@@{
{
while (atomicCAS(&lock, 0, 1) != 0)
__threadfence();
data[1]++;
__threadfence();
atomicExch(&lock, 0);
data[0]++;
}                                           // @@@}

int main() 
{
    int N = 5;
    unsigned int *d_data;
    cudaMalloc(&d_data, N*sizeof(unsigned int));
    unsigned int *t = (unsigned int*)malloc(N*sizeof(unsigned int));
    for (int i = 0; i < N; ++i)
       t[i] = 0;
    cudaMemcpy(d_data, t, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    kmain<<<NBLOCKS,TPERBLK>>>(d_data);
    cudaMemcpy(t, d_data, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("%u, %u\n", t[0], t[1]);
    errCheck();
    return 0;
}
