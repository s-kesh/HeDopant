#include "rk4_opencl.hpp"
#include "opencl_context.hpp"

#ifdef HAS_ARROW
#include "arrow_io.hpp"
#endif

#include <iostream>
#include <clblast.h>

#define CL_CHECK(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL Error: " << err \
                        << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// The OpenCL kernel source (must be a C string or loaded from a file)
const char* rk4_kernels_source = R"OPENCL_KERNEL(
// OpenCL C kernels
__kernel void kernel_compute_dia(
    const double pressure,
    __global const double* restrict alpha,
    __global const double* restrict x_in,
    __global double* restrict d,
    const int total)
{
    int idx = get_global_id(0);
    if (idx >= total) return;
    d[idx] = pressure * alpha[idx] * x_in[idx];
}

// Column-major layout: idx = i + j * N
__kernel void kernel_compute_diff(
    __global const double* restrict d,
    __global double* restrict x_out,
    const int N,
    const int K)
{
    int idx = get_global_id(0);
    const int total = N * K;
    if (idx >= total) return;

    int i = idx % N;      // row
    int j = idx / N;      // column

    if (i == 0) { // first row
        x_out[idx] = -d[idx];
    } else {
        x_out[idx] = d[idx - 1] - d[idx];
    }
}
)OPENCL_KERNEL";

void copy_device_host(double* dist, cl::Buffer& src_buffer, const int size, cl::CommandQueue& queue) {
    CL_CHECK(queue.enqueueReadBuffer(src_buffer, CL_TRUE, 0, size * sizeof(double), dist));

    for (int i = 0; i < 10 && i < size; ++i) {
        std::cout << dist[i] << " ";
    }
    std::cout << std::endl;
}


void RK4Backend_OpenCL::solve_ode(
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

    OpenCLContext cl_ctx(rk4_kernels_source);
    cl::Context& context = cl_ctx.getContext();
    cl::CommandQueue& queue = cl_ctx.getQueue();

    // Compile and build the kernel program
    cl::Program& program = cl_ctx.getProgram();

    // Create kernels
    cl::Kernel kernel_compute_dia(program, "kernel_compute_dia");
    cl::Kernel kernel_compute_diff(program, "kernel_compute_diff");

    // Device Memory Allocation
    cl::Buffer d_y, d_alpha, d_temp, d_dia, d_kfinal;
    d_y      = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(double));
    d_alpha  = cl::Buffer(context, CL_MEM_READ_ONLY,  size * sizeof(double));
    d_temp   = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(double));
    d_dia    = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(double));
    d_kfinal = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(double));

    // Copy Host to Device
    CL_CHECK(queue.enqueueWriteBuffer(d_y, CL_TRUE, 0, size * sizeof(double), y));
    CL_CHECK(queue.enqueueWriteBuffer(d_alpha, CL_TRUE, 0, size * sizeof(double), alpha));

    // CLBlast constants (equivalent to cuBLAS constants)
    const double one = 1.0;
    const double two = 2.0;
    const double step = step_size;
    const double half_step = 0.5 * step_size;
    const double sixth_step = step / 6.0;

    // Grid and Block setup
    const size_t threads_per_block = rows;
    const size_t global_size = (size + threads_per_block - 1) / threads_per_block * threads_per_block;
    cl::NDRange global(global_size);
    cl::NDRange local(threads_per_block);

    float progress = 0.0f;
    print_progress_bar(progress);

    // CLBlast setup
    clblast::StatusCode clblas_status;
    cl::Event event;

    for (std::size_t step_idx = 0; step_idx < number_of_steps; ++step_idx) {

        // k1
        // d_temp = y
        clblas_status = clblast::Copy<double>(size, d_y(), 0, 1, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Copy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_dia = pressure * alpha * d_temp (kernel_compute_dia)
        kernel_compute_dia.setArg(0, pressure);
        kernel_compute_dia.setArg(1, d_alpha);
        kernel_compute_dia.setArg(2, d_temp);
        kernel_compute_dia.setArg(3, d_dia);
        kernel_compute_dia.setArg(4, size);
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_dia, cl::NullRange, global, local));
        CL_CHECK(queue.finish()); // Ensure kernel completes before next step

        // d_temp = compute_diff(d_dia) = k1 (kernel_compute_diff)
        kernel_compute_diff.setArg(0, d_dia);
        kernel_compute_diff.setArg(1, d_temp);
        kernel_compute_diff.setArg(2, rows);
        kernel_compute_diff.setArg(3, cols);
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_diff, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // kfinal = k1
        clblas_status = clblast::Copy<double>(size, d_temp(), 0, 1, d_kfinal(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Copy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // k2
        // d_temp = d_temp * (h/2) (cublasDscal -> CLBlast DSCAL)
        clblas_status = clblast::Scal<double>(size, half_step, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Scal: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_temp = d_temp + y -> now d_temp = y + h/2 * k1
        clblas_status = clblast::Axpy<double>(size, one, d_y(), 0, 1, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_dia = pressure * alpha * d_temp (kernel_compute_dia)
        kernel_compute_dia.setArg(2, d_temp);
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_dia, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // d_temp = compute_diff(d_dia) (kernel_compute_diff)
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_diff, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // kfinal += 2 * k2
        clblas_status = clblast::Axpy<double>(size, two, d_temp(), 0, 1, d_kfinal(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // k3
        // d_temp = d_temp * (h/2)
        clblas_status = clblast::Scal<double>(size, half_step, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Scal: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_temp = d_temp + y -> now d_temp = y + h/2 * k2
        clblas_status = clblast::Axpy<double>(size, one, d_y(), 0, 1, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_dia = pressure * alpha * d_temp (kernel_compute_dia)
        kernel_compute_dia.setArg(2, d_temp);
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_dia, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // d_temp = compute_diff(d_dia) (kernel_compute_diff)
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_diff, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // kfinal += 2 * k3
        clblas_status = clblast::Axpy<double>(size, two, d_temp(), 0, 1, d_kfinal(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // k4
        // d_temp = d_temp * h
        clblas_status = clblast::Scal<double>(size, step, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Scal: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_temp = d_temp + y -> now d_temp = y + h * k3
        clblas_status = clblast::Axpy<double>(size, one, d_y(), 0, 1, d_temp(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // d_dia = pressure * alpha * d_temp (kernel_compute_dia)
        kernel_compute_dia.setArg(2, d_temp);
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_dia, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // d_temp = compute_diff(d_dia) (kernel_compute_diff)
        CL_CHECK(queue.enqueueNDRangeKernel(kernel_compute_diff, cl::NullRange, global, local));
        CL_CHECK(queue.finish());

        // kfinal += k4
        clblas_status = clblast::Axpy<double>(size, one, d_temp(), 0, 1, d_kfinal(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // y = y + (h/6) * kfinal
        clblas_status = clblast::Axpy<double>(size, sixth_step, d_kfinal(), 0, 1, d_y(), 0, 1, &queue(), &event());
        if (clblas_status != clblast::StatusCode::kSuccess) {
            std::cerr << "Error in clblast::Axpy: " << (int)clblas_status << std::endl;
        }
        event.wait();

        // progress
        progress = static_cast<float>(step_idx + 1) / static_cast<float>(number_of_steps);
        if ((step_idx + 1) % (std::max<std::size_t>(1, number_of_steps / 100)) == 0) {
            print_progress_bar(progress);
        }

        if (trajectory) {
            // copy back current y (bytes)
            CL_CHECK(queue.enqueueReadBuffer(d_y, CL_TRUE, 0, size * sizeof(double), y));
            #ifdef HAS_ARROW
            arrow_io.write_step(step_idx + 1, rows, cols, y);
            #endif
        }
    }

    print_progress_bar(1.0f);
    std::cout << std::endl;

    #ifdef HAS_ARROW
    arrow_io.close()
    #endif

    // Move data back to host
    CL_CHECK(queue.enqueueReadBuffer(d_y, CL_TRUE, 0, size * sizeof(double), y));
}
