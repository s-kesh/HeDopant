# HeDopant: Helium Droplet Doping Simulation

## Overview

HeDopant is a simulation tool for modeling the evolution of dopant cluster-size distributions as helium nanodroplets traverse a doping cell. The simulation integrates the dopant pick-up process using a fourth-order Runge–Kutta (RK4) solver and supports both CPU and GPU execution.

## Features

- **Flexible Backend**: Supports both CPU and GPU (via CUDA or OpenCL) for RK4 solver.
- **Configuration**: Allow configuration through a simple YAML file, but allow overriding the configuration options using the command line. A sample YAML file is provided in the `data` folder.
- **Dopant Support**: Dopant species are configured using YAML files provided in the `data` folder. Just add your own dopant YAML file and specify it in config.yaml.
- **Output**: Final distribution of dopant size would be written on the computer. By configuring the flag `trajectory` to `true`, the distribution would be saved at each RK4 step.

## Building

This project uses Meson Build System.

### Prerequisited

- C++23 compiler (e.g., `g++`)  
- CUDA (for GPU support)
- [Meson](https://mesonbuild.com/) build system  
- Dependencies: `eigen`, `yaml-cpp`, `arrow.`

```bash
meson setup build
meson compile -C build
```

## Usage

1. Edit `config.yaml` to specify your simulation parameters:

```yaml
name: "Helium Droplet"
number_of_atoms: [10000, 19000]
type: "LOGNORMAL" # options are: "NONE", "LOGNORMAL", or "EXPONENTIAL"
dopant: "krypton"
doping_cell: 0.018 # m
dopant_pressure: 1E-4
max_dopant: 100
rk_steps: 100000
datadir: "../data"
output: "he_drop_krypton"
trajectory: false
```
**Note:**

- Adding multiple `number_of_atoms` does not make the simulation slow. The only cost increase is the final weighted averaging over the droplet distribution.
- Biggest performance hit comes with setting `trajectory` to `true`.
  1. Writing to disk is slower than doing calculations.
  2. In the case of GPU acceleration, it needs to copy the dopant distribution from GPU memory to CPU memory, which is very slow.
  
2. Run the program

```bash
./build/HeDopant --config config.yaml
```
- --config <file>: Specify an alternate configuration file
- --help: Show usage information

You can override any configuration option at the command line.

```bash
./build/HeDopant --config config.yaml --dopant water --dopant_pressure 1.5
```

## Output

The directory of output files can be controlled either by setting the option `output` in the config file or by the command line option `--output <output>`.

The evolved distribution would be saved as `<output>/dist_k.txt` and `<output>/dist_Nk.txt`.

A few additional output files will also be generated to ensure everything is working smoothly.

- `<output>/vcluster_ebe.txt`: Velocity and binding energy of He droplet for a certain size. It is interpolated using the `droplet.txt` file in the `data` directory.
- `<output>/He_<N>_size_distribution.txt`: Size distribution of the droplet with mean size `<N>`.
- `<output>_evap.txt`: For each initial size in droplet distribution, we will have `alpha` parameter and number of remaining He atoms after absorption of `k` dopants.
- `<output>/final_y_<no>.txt`: Final dopant distribution without taking droplet distribution into account. It would be the same as the `output.txt` file if you select the distribution type `NONE`.
- `<output>/trajectory_<no>.arrow`: Binary file having dopant distribution at each RK4 step. To read this file in a pandas dataframe a simple python script is provided in `utils` directory.
