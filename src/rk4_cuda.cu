#include "rk4_cuda.hpp"

#ifdef HAS_ARROW
#include "arrow_io.hpp"
#endif

#include <cstdlib>
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

// Computes (pressure * alpha * y)_prev - (pressure * alpha * y)_curr
__global__ void kernel_compute(
    const double pressure,
    const double* __restrict__ alpha,
    const double* __restrict__ y,
    const double* __restrict__ k_prev,
    const double scale,
    double* __restrict__ k_out,
    const bool k1_flag,
    const double weight,
    double* __restrict__ k_sum,
    const int rows,
    const int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = col * rows + row;
        double curr = y[idx];
        if (scale != 0.0) curr += scale * k_prev[idx];
        double val_curr = pressure * alpha[idx] * curr;
        if (row == 0) {
            k_out[idx] = -val_curr;
        } else {
            double prev = y[idx - 1];
            if (scale != 0.0) prev += scale * k_prev[idx - 1];
            double val_prev = pressure * alpha[idx - 1] * prev;
            k_out[idx] = val_prev - val_curr;
        }

        if (k1_flag) {
            k_sum[idx] = k_out[idx];
        } else {
            k_sum[idx] += weight * k_out[idx];
        }
    }
}

void copy_device_host(double* dist, const double* src, const int size) {
    CUDA_CHECK(cudaMemcpy(dist, src, size * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10; ++i) {
        std::cout << dist[i] << " ";
    }
    std::cout << std::endl;
}

void RK4Backend_CUDA::solve_ode(
    const std::size_t number_of_steps,
    const double step_size,
    const double pressure,
    const int rows,
    const int cols,
    const double *alpha,
    const bool trajectory,
    const std::string& output_file,
    double *y
) {

    const int size = rows * cols;
    #ifdef HAS_ARROW
    ArrowIO arrow_io(output_file);
    #endif

    if (trajectory) {
        #ifdef HAS_ARROW
        arrow_io.write_step(0, rows, cols, y);
        #endif
    }

    double *d_y = nullptr, *d_alpha = nullptr, *d_temp1 = nullptr, *d_temp2 = nullptr, *d_kfinal = nullptr;

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_y, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp1, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp2, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_kfinal, size * sizeof(double)));

    // copy host arrays to device
    CUDA_CHECK(cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha, size * sizeof(double), cudaMemcpyHostToDevice));

    // cuBLAS helpers
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double one = 1.0;
    const double two = 2.0;
    const double step = step_size;
    const double half_step = 0.5 * step_size;
    const double sixth_step = step / 6.0;

    // Launch Config
    const int threads_per_block = 256;

    dim3 block(threads_per_block, 1);
    dim3 grid((rows + threads_per_block - 1) / threads_per_block, cols);

    // Debug
    std::cout << "Matrix Rows " << rows << " and Columns " << cols << std::endl;

    print_progress_bar(0, number_of_steps);

    for (std::size_t step_idx = 0; step_idx < number_of_steps; ++step_idx) {

        // k1 = f(y)
        kernel_compute<<<grid, block>>>(pressure, d_alpha, d_y, nullptr, 0.0, d_temp1, true, 0.0, d_kfinal, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        // CUBLAS_CHECK(cublasDcopy(handle, size, d_temp1, 1, d_kfinal, 1));

        // k2 = f( y + h/2 * k1)
        kernel_compute<<<grid, block>>>(pressure, d_alpha, d_y, d_temp1, half_step, d_temp2, false, two, d_kfinal, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        // CUBLAS_CHECK(cublasDaxpy(handle, size, &two, d_temp2, 1, d_kfinal, 1));

        // k3 = f(y + h/2 * k2)
        kernel_compute<<<grid, block>>>(pressure, d_alpha, d_y, d_temp2, half_step, d_temp1, false, two, d_kfinal, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        // CUBLAS_CHECK(cublasDaxpy(handle, size, &two, d_temp1, 1, d_kfinal, 1));

        // k4 = f(y + h * k3)
        kernel_compute<<<grid, block>>>(pressure, d_alpha, d_y, d_temp1, step, d_temp2, false, one, d_kfinal, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        // CUBLAS_CHECK(cublasDaxpy(handle, size, &one, d_temp2, 1, d_kfinal, 1));

        // y = y + (h/6) * kfinal
        CUBLAS_CHECK(cublasDaxpy(handle, size, &sixth_step, d_kfinal, 1, d_y, 1));

        // progress
        // progress = static_cast<float>(step_idx + 1) / static_cast<float>(number_of_steps);
        // if ((step_idx + 1) % (std::max<std::size_t>(1, number_of_steps / 100)) == 0) {
        //     print_progress_bar(progress);
        // }
        // if ((step_idx + 1) % (std::max<std::size_t>(1, number_of_steps / 1000)) == 0)
            print_progress_bar(step_idx + 1, number_of_steps);

        #ifdef HAS_ARROW
        if (trajectory) {
            // copy back current y (bytes)
            CUDA_CHECK(cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost));
            arrow_io.write_step(step_idx + 1, rows, cols, y);
        }
        #endif
    }

    // print_progress_bar(1.0f);

    print_progress_bar(number_of_steps, number_of_steps);
    std::cout << std::endl;

    #ifdef HAS_ARROW
    arrow_io.close();
    #endif

    // Move data back to host (bytes)
    CUDA_CHECK(cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory and destroy handle
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_temp1));
    CUDA_CHECK(cudaFree(d_kfinal));
    CUDA_CHECK(cudaFree(d_temp2));
    CUDA_CHECK(cudaFree(d_alpha));

    CUBLAS_CHECK(cublasDestroy(handle));
}
