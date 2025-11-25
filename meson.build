project(
  'HeDopant',
  'cpp',
  version: '1.0',
  default_options: [
    'cpp_std=c++23',
    'warning_level=3',
  ],
)

backend_opt = get_option('rk4backend')

bt = get_option('buildtype')
if bt == 'debugoptimized'
  add_project_arguments(['-pg'], language: 'cpp')
  add_project_link_arguments(['-pg'], language: 'cpp')
endif

# Include directory
inc = include_directories('header')

# Dependencies
yaml_dep = dependency('yaml-cpp')
arrow_dep = dependency('arrow')
eigen_dep = dependency('eigen3')

# CUDA Dependencies
cuda_dep = dependency(
  'cuda',
  version: '>12.0',
  required: false,
  modules: ['cublas'],
)

opencl_dep = dependency(
  'OpenCL',
  required: false,
)
clblast_dep = dependency(
  'clblast',
  required: false,
)

# Default to CPU.
selected_backend = 'cpu'

if backend_opt == 'cuda'
  if cuda_dep.found()
    selected_backend = 'cuda'
    message('CUDA backend explicitly selected.')
  else
    error('CUDA backend was forced but dependencies (CUDA/cuBLAS) were not found.')
  endif
elif backend_opt == 'opencl'
  if opencl_dep.found() and clblast_dep.found()
    selected_backend = 'opencl'
    message('OpenCL backend explicitly selected.')
  else
    error(
      'OpenCL backend was forced but dependencies (OpenCL/CLBlast) were not found.',
    )
  endif
elif backend_opt == 'auto'
  if cuda_dep.found()
    selected_backend = 'cuda'
    message('CUDA detected. Backend set to CUDA (auto mode).')
  elif opencl_dep.found() and clblast_dep.found()
    selected_backend = 'opencl'
    message('OpenCL/CLBlast detected. Backend set to OpenCL (auto mode).')
  else
    message('No accelerator detected. Backend set to CPU (auto mode).')
  endif
elif backend_opt == 'cpu'
  message('CPU backend explicitly selected.')
endif

# Base source files
srcs = files(
  'src/droplet.cpp',
  'src/interpolate.cpp',
  'src/main.cpp',
)

project_dependencies = [yaml_dep, arrow_dep, eigen_dep]

if selected_backend == 'cuda'
  add_languages('cuda')
  add_project_arguments(['-DRK4_HAS_GPU', '-DRK4_BACKEND_CUDA'], language: 'cpp')
  add_project_arguments(['-DCUDA_STD=c++17'], language: 'cuda')
  srcs += files('src/rk4_cuda.cu')
  project_dependencies += [cuda_dep]

elif selected_backend == 'opencl'
  add_project_arguments(['-DRK4_HAS_GPU', '-DRK4_BACKEND_OPENCL'], language: 'cpp')
  srcs += files('src/rk4_opencl.cpp')
  project_dependencies += [opencl_dep, clblast_dep]

else
  srcs += files('src/rk4_cpu.cpp')
endif

# Build Executable

executable(
  'HeDopant',
  srcs,
  include_directories: inc,
  dependencies: project_dependencies,
  install: false,
)
