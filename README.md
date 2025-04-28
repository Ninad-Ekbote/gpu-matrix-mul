# gpu-matrix-mul
Overview
This project demonstrates matrix multiplication using CUDA, optimizing the process with a tiled matrix multiplication technique. It showcases how to use shared memory to improve memory access efficiency and achieve faster computation on the GPU. The program multiplies two square matrices and compares the GPU results with the CPU results to validate correctness.

Key Features
Tiled Matrix Multiplication: Using shared memory to hold sub-matrices for each block to improve memory locality.

CPU vs GPU Comparison: The results from the GPU are compared with those computed on the CPU to ensure correctness.

CUDA Error Handling: Uses helper functions for CUDA error checking to simplify the development process.

Prerequisites
Before running the program, ensure you have the following installed:

CUDA Toolkit: Ensure that you have a working installation of CUDA. The version should be compatible with your hardware and CUDA runtime.

NVIDIA GPU: A compatible NVIDIA GPU with CUDA capability is required to run the program on the GPU.

Files
CPU_GLOBAL_SHARED_MATRIX.cu: The main CUDA code for matrix multiplication using the tiled approach.

helper_functions.h and helper_cuda.h: Utility headers for common CUDA functions, including error checking and initialization.
