#include "rk4_backend.hpp"

#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << err \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void kernel_compute_dia(
    const double pressure,
    const double* __restrict__ alpha,
    const double* __restrict__ x_in,
    double* __restrict__ d,
    const int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    d[idx] = pressure * alpha[idx] * x_in[idx];
}

// Column-major layout: idx = i + j * N
__global__ void kernel_compute_diff(
    const double* __restrict__ d,
    double* __restrict__ x_out,
    const int N,
    const int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    if (idx >= total) return;

    int i = idx % N;     // row
    int j = idx / N;     // column (0..K-1)

    if (j == 0) {
        x_out[idx] = -d[idx];
    } else {
        // idx_prev = i + (j-1)*N
        x_out[idx] = d[i + (j - 1) * N] - d[idx];
    }
}

void RK4Backend_GPU::solve_ode(
    const std::size_t number_of_steps,
    const double step_size,
    const double pressure,
    const int rows,
    const int cols,
    const double *alpha,
    // const bool trajectory,
    // ArrowIO& arrow_io,
    double *y
) {

    const int size = rows * cols;

    double *d_y = nullptr, *d_alpha = nullptr, *d_temp = nullptr, *d_dia = nullptr, *d_kfinal = nullptr;

    // allocate device memory (use CUDA_CHECK)
    CUDA_CHECK(cudaMalloc(&d_y, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dia, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_kfinal, size * sizeof(double)));

    // copy host arrays to device (use bytes)
    CUDA_CHECK(cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha, size * sizeof(double), cudaMemcpyHostToDevice));

    // cuBLAS helpers
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate_v2(&handle));

    const double one = 1.0;
    const double two = 2.0;
    const double one_sixth = 1.0 / 6.0;
    const double half_step = 0.5 * step_size;
    const double step = step_size;

    int threads_per_block = 256;
    int blocks_per_grid = static_cast<int>((size + threads_per_block - 1) / threads_per_block);

    float progress = 0.0f;

    for (std::size_t step_idx = 0; step_idx < number_of_steps; ++step_idx) {

        // ---------- k1 ----------
        // d_temp = y
        CUBLAS_CHECK(cublasDcopy_v2(handle, size, d_y, 1, d_temp, 1));

        // d_dia = pressure * alpha * d_temp
        kernel_compute_dia<<<blocks_per_grid, threads_per_block>>>(pressure, d_alpha, d_temp, d_dia, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // d_temp = compute_diff(d_dia)  (reuses d_temp)
        kernel_compute_diff<<<blocks_per_grid, threads_per_block>>>(d_dia, d_temp, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // kfinal = k1
        CUBLAS_CHECK(cublasDcopy_v2(handle, size, d_temp, 1, d_kfinal, 1));

        // ---------- k2 ----------
        // d_temp = d_temp * (h/2)   (d_temp currently holds k1)
        CUBLAS_CHECK(cublasDscal_v2(handle, size, &half_step, d_temp, 1));
        // d_temp = d_temp + y       -> now d_temp = y + h/2 * k1
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &one, d_y, 1, d_temp, 1));

        kernel_compute_dia<<<blocks_per_grid, threads_per_block>>>(pressure, d_alpha, d_temp, d_dia, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        kernel_compute_diff<<<blocks_per_grid, threads_per_block>>>(d_dia, d_temp, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // kfinal += 2 * k2  (note: d_temp holds k2)
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &two, d_temp, 1, d_kfinal, 1));

        // ---------- k3 ----------
        // d_temp = d_temp * (h/2)  (d_temp currently holds k2)
        CUBLAS_CHECK(cublasDscal_v2(handle, size, &half_step, d_temp, 1));
        // d_temp = d_temp + y      -> now d_temp = y + h/2 * k2
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &one, d_y, 1, d_temp, 1));

        kernel_compute_dia<<<blocks_per_grid, threads_per_block>>>(pressure, d_alpha, d_temp, d_dia, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        kernel_compute_diff<<<blocks_per_grid, threads_per_block>>>(d_dia, d_temp, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // kfinal += 2 * k3  (d_temp holds k3)
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &two, d_temp, 1, d_kfinal, 1));

        // ---------- k4 ----------
        // d_temp = d_temp * h   (d_temp currently holds k3)
        CUBLAS_CHECK(cublasDscal_v2(handle, size, &step, d_temp, 1));
        // d_temp = d_temp + y   -> now d_temp = y + h * k3
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &one, d_y, 1, d_temp, 1));

        kernel_compute_dia<<<blocks_per_grid, threads_per_block>>>(pressure, d_alpha, d_temp, d_dia, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        kernel_compute_diff<<<blocks_per_grid, threads_per_block>>>(d_dia, d_temp, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // kfinal += k4  (d_temp holds k4)
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &one, d_temp, 1, d_kfinal, 1));

        // ---------- update y ----------
        // y = y + (1/6) * kfinal
        CUBLAS_CHECK(cublasDaxpy_v2(handle, size, &one_sixth, d_kfinal, 1, d_y, 1));

        // progress
        progress = static_cast<float>(step_idx + 1) / static_cast<float>(number_of_steps);
        if ((step_idx + 1) % (std::max<std::size_t>(1, number_of_steps / 100)) == 0) {
            print_progress_bar(progress);
        }

        // if (trajectory) {
        //     // copy back current y (bytes)
        //     CUDA_CHECK(cudaMemcpy(y.data(), d_y, size * sizeof(double), cudaMemcpyDeviceToHost));
        //     arrow_io.write_step(step_idx, size, y.data());
        // }
    }

    print_progress_bar(1.0f);
    std::cout << std::endl;

    // Move data back to host (bytes)
    CUDA_CHECK(cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory and destroy handle
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_kfinal));
    CUDA_CHECK(cudaFree(d_dia));
    CUDA_CHECK(cudaFree(d_alpha));

    CUBLAS_CHECK(cublasDestroy_v2(handle));
}
