project(
  'HeDopant',
  'cpp',
  version: '1.0',
  default_options: [
    'cpp_std=c++23',
    'warning_level=3',
  ],
)

#  Determine build-type flags
bt = get_option('buildtype')
if bt == 'debugoptimized'
  add_project_arguments(['-pg'], language: 'cpp')
  add_project_link_arguments(['-pg'], language: 'cpp')
endif

#  Include directory
inc = include_directories('header')

#  Dependencies
yaml_dep = dependency('yaml-cpp')
arrow_dep = dependency('arrow')
eigen_dep = dependency('eigen3')
omp_dep = dependency('openmp', required: false)

#  Source files
srcs = files(
  'src/droplet.cpp',
  'src/interpolate.cpp',
  'src/main.cpp',
  'src/rk4_cpu.cpp',
)

#  Build executable
executable(
  'HeDopant',
  srcs,
  include_directories: inc,
  dependencies: [yaml_dep, arrow_dep, eigen_dep, omp_dep],
  install: false,
)
