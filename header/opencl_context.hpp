#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

class OpenCLContext {
private:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    // Helper to check and handle errors
    void checkError(cl_int err, const std::string& message) {
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL Error: " << err << " (" << message << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

public:
    OpenCLContext(const std::string& kernel_source) {
        cl_int err;

        // Get Platforms
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        checkError(err, "Platform::get");
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found." << std::endl;
            exit(EXIT_FAILURE);
        }

        platform = platforms[0];

        // Get Devices
        std::vector<cl::Device> devices;
        err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (err != CL_SUCCESS || devices.empty()) {
            err = platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            checkError(err, "Device::get");
        }
        device = devices[0];

        // Create Context
        context = cl::Context(device, nullptr, nullptr, nullptr, &err);
        checkError(err, "Context::Context");

        // Create Command Queue
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "CommandQueue::CommandQueue");

        std::vector<std::string> sources = {kernel_source};
        program = cl::Program(context, sources, &err);
        checkError(err, "Program::Program");

        err = program.build({device});
        if (err != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "OpenCL Program Build Error on device " << device.getInfo<CL_DEVICE_NAME>() << ":\n" << build_log << std::endl;
            // Print error and then force exit
            checkError(err, "Program::build");
        }
    }

    // Accessors for resource handles
    cl::Context& getContext() { return context; }
    cl::CommandQueue& getQueue() { return queue; }
    cl::Program& getProgram() { return program; }
    cl::Device& getDevice() { return device; }
};
