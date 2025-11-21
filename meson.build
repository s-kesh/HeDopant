project(
  'HeDopant',
  'cpp',
  version: '1.0',
  default_options: [
    'cpp_std=c++23',
    'cuda_std=c++17',
    'warning_level=3',
  ],
)

# Determine build-type flags
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

# GPU options
gpu_opt = get_option('gpu')

cuda_dep = dependency(
  'cuda',
  version: '>12.0',
  required: false,
  modules: ['cublas'],
)

has_gpu = true

if gpu_opt == 'disabled'
  message('GPU support explicitly disabled by user.')
elif gpu_opt == 'enabled'
  if cuda_dep.found()
    has_gpu = true
    message('GPU support explicitly enabled.')
  else
    error('GPU was forced enabled (-Dgpu=enabled) but CUDA was not found!')
  endif
else
  # auto
  if cuda_dep.found()
    has_gpu = true
    message('CUDA detected. GPU enabled (auto mode).')
  else
    message('CUDA not found. GPU disabled (auto mode).')
  endif
endif
# ------------------------------------

# Source files
srcs = files(
  'src/droplet.cpp',
  'src/interpolate.cpp',
  'src/main.cpp',
  'src/rk4_cpu.cpp',
)

project_dependencies = [yaml_dep, arrow_dep, eigen_dep]

# Conditionally add GPU source and define flag
if has_gpu
  add_languages('cuda')
  srcs += files('src/rk4_gpu.cu')
  add_project_arguments(['-DRK4_HAS_GPU'], language: 'cpp')
  project_dependencies += [cuda_dep]
endif

# Build executable
executable(
  'HeDopant',
  srcs,
  include_directories: inc,
  dependencies: project_dependencies,
  install: false,
)
