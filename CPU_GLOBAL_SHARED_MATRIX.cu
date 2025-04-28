
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <stdio.h>

#define TILE_WIDTH 3
void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF ) / 10.0f;
    }

    return;
}

void MulMatrixOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {

        for (int j = 0; j < N; j++)
        {
            
            float value = 0;
            for (int k = 0; k < N; k++) {
                value += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = value;
        }



    }
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 0.1;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("%d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("PASS\n\n");
    else
        printf("FAIL\n\n");
}



__global__ void tiledmatrixMultiplyGPU(float* MatA, float* MatB, float* MatC, int N)
{
    int bx = blockIdx.y;
    int by = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * by + ty;
    int j = blockDim.x * bx + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];


    float value = 0;

    for (int phase = 0; phase < (N + TILE_WIDTH - 1) / TILE_WIDTH; phase++) {
        if (i < N && (phase * TILE_WIDTH + tx) < N)
            sh_A[ty][tx] = MatA[i * N + phase * TILE_WIDTH + tx];
        else
            sh_A[ty][tx] = 0;

        if ((phase * TILE_WIDTH + ty) < N && j < N)
            sh_B[ty][tx] = MatB[(phase * TILE_WIDTH + ty) * N + j];
        else
            sh_B[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __syncthreads();
    }

    if (i < N && j < N)
    {
        MatC[i * N + j] = value;
    }

}


/*
__global__ void matrixMultiplyGPU(float* MatA, float* MatB, float* MatC, int N)
{
 
unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = row * N + col;

if ((row < N) && (col < N)) // As long as your code prevents access violation, you can modify this "if" condition.
{
    float value = 0;

    for (int k = 0; k < N; k++) {

        value += MatA[row * N + k] * MatB[k * N + col];

    }

    MatC[idx] = value;
}

}

*/


int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    // set up data size of matrix
    int N = 5;
    

    int nxy = N * N;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", N, N);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    for (int row = 0; row < N; row++) {

        for (int col = 0; col < N; col++) {
            printf("%f\t", h_A[row * N + col]);
        }
        printf("\n");
    }


    printf("\n");
    for (int row = 0; row < N; row++) {

        for (int col = 0; col < N; col++) {
            printf("%f\t", h_B[row * N + col]);
        }
        printf("\n");
    }

    printf("\n");


    // add matrix at host side for result checks
    MulMatrixOnHost(h_A, h_B, hostRef, N);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    checkCudaErrors(cudaMalloc((void **)&d_MatA, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatB, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatC, nBytes));

    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    int dimx = 3;
    int dimy = 3;
    dim3 block(dimx, dimy);
    dim3 grid((N+ dimx-1)/dimx,(N + dimy - 1)/ dimy);
    checkCudaErrors(cudaEventRecord(start, 0));
    tiledmatrixMultiplyGPU <<<grid, block >>>(d_MatA, d_MatB, d_MatC, N);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    

    float kernel_time;
    checkCudaErrors(cudaEventElapsedTime(&kernel_time, start, stop));

    // checkCudaErrors kernel error
    checkCudaErrors(cudaGetLastError());

    // copy kernel result back to host side
    checkCudaErrors(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N*N; i++)
    {
       
            
            printf("%d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
           
        
    }
    

    // checkCudaErrors device results
    checkResult(hostRef, gpuRef, nxy);

    printf(" Kernel execution time\t\t\t: %f ms\n",
        kernel_time);
        
	  printf("Name :- Ninad Ekbote, PID:- A69026968\n");

    // free device global memory
    checkCudaErrors(cudaFree(d_MatA));
    checkCudaErrors(cudaFree(d_MatB));
    checkCudaErrors(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    checkCudaErrors(cudaDeviceReset());

    return (0);
}
